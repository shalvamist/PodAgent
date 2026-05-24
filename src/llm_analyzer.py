"""LLM analyzer module — feeds structured transcripts to local LLMs (Ollama/LM Studio)."""

import json
import os
import time
import logging
import httpx
import re
from dataclasses import dataclass
from typing import Optional, Literal

from src import folder_manager

logger = logging.getLogger(__name__)

# Analysis mode types
AnalysisMode = Literal["summary", "insights", "notes", "blog"]


@dataclass
class LLMAnalysisResult:
    """Result from LLM analysis of a transcript."""
    analysis_mode: str
    llm_model: str
    provider: str  # "ollama" or "lmstudio"
    summary_text: str  # The main output text
    structured_output: Optional[dict]  # JSON-structured output if requested
    raw_response: str  # Full raw response from LLM
    processing_time_seconds: float
    transcript_id: str  # podcast/video_id used for analysis
    output_path: Optional[str]  # Path to saved analysis file


@dataclass
class LLMAnalyzerConfig:
    """Configuration for LLM analyzer."""
    provider: str = "ollama"  # "ollama" or "lmstudio"
    model: str = "llama3"
    base_url: str = "http://localhost:11434"  # Ollama default
    lmstudio_url: str = "http://localhost:1234"  # LM Studio default
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: int = 300
    streaming: bool = True
    enable_structured_output: bool = True  # Request JSON output when possible
    # Mode-specific token limits (summary/insights don't need full 4096)
    max_tokens_by_mode: dict = None  # Will be set in __init__


class LLMAnalyzer:
    """Analyze podcast transcripts using local LLMs (Ollama or LM Studio)."""

    def __init__(self, config: Optional[LLMAnalyzerConfig] = None):
        self.config = config or LLMAnalyzerConfig()
        # Set mode-specific token limits
        self.config.max_tokens_by_mode = {
            "summary": 2048,     # Short summary + structured JSON
            "insights": 2048,   # Bullet points + structured JSON
            "notes": 2048,      # Structured notes + structured JSON
            "blog": 3072,       # Blog post + structured JSON
        }
        self.client = httpx.Client(
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        )
        # Result cache: keyed by (video_id, mode) — skip re-analyzing same combo
        self._cache: dict[tuple[str, str], LLMAnalysisResult] = {}
        self._cache_max_size = 50  # LRU cap
        logger.info(f"LLMAnalyzer initialized: provider={self.config.provider}, model={self.config.model}")

    def _get_api_url(self) -> str:
        """Get the correct API endpoint URL based on provider."""
        if self.config.provider == "ollama":
            return f"{self.config.base_url}/api/generate"
        elif self.config.provider == "lmstudio":
            return f"{self.config.lmstudio_url}/v1/completions"
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _evict_cache(self):
        """Evict oldest entries if cache exceeds max size (simple FIFO)."""
        if len(self._cache) > self._cache_max_size:
            # Pop the first inserted key (oldest)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted: {oldest_key} (size now {len(self._cache)})")

    def _build_prompt(self, transcript: dict, mode: AnalysisMode) -> str:
        """Build the prompt for LLM analysis based on mode and transcript content.

        Uses segments (with timestamps and speaker labels) instead of raw_text
        for better structure and fewer wasted tokens.
        """
        title = transcript.get("video_title", "Unknown")
        speakers = transcript.get("speakers", [])
        segments = transcript.get("segments", [])

        speaker_info = ""
        if speakers:
            speaker_info = "\nSpeakers:\n"
            for sp in speakers:
                label = sp.get("label", "Unknown")
                speaker_info += f"  - {label}\n"

        # Use segments for structured input — much more efficient than raw_text
        # Limit to first 200 segments for context, then summarize the rest
        max_segments = 200
        if len(segments) > max_segments:
            segment_text = ""
            for seg in segments[:max_segments]:
                speaker = seg.get("speaker_label", "Unknown")
                text = seg.get("text", "")
                segment_text += f"[{speaker}] {text}\n"
            segment_text += f"\n\n[... {len(segments) - max_segments} more segments truncated ...]\n"
        else:
            segment_text = ""
            for seg in segments:
                speaker = seg.get("speaker_label", "Unknown")
                text = seg.get("text", "")
                segment_text += f"[{speaker}] {text}\n"

        prompts = {
            "summary": f"""Analyze the following podcast transcript and provide a concise summary.

Podcast Title: {title}
{speaker_info}

Transcript (key excerpts with speaker labels):
{segment_text}

Provide a 1-2 paragraph summary that captures:
1. The main topics discussed
2. Key points and arguments
3. Any conclusions or recommendations
4. Notable quotes or insights

Keep it concise and informative. Output as plain text.""",

            "insights": f"""Extract key insights from the following podcast transcript.

Podcast Title: {title}
{speaker_info}

Transcript (key excerpts with speaker labels):
{segment_text}

Provide a structured list of key insights:
1. Main topics covered (with brief explanation)
2. Key arguments or positions taken
3. Notable data points or statistics mentioned
4. Action items or recommendations
5. Unexpected or surprising points

Format as bullet points with clear headings. Output as plain text.""",

            "notes": f"""Generate structured notes from the following podcast transcript.

Podcast Title: {title}
{speaker_info}

Transcript (key excerpts with speaker labels):
{segment_text}

Create structured notes with:
1. Executive summary (2-3 sentences)
2. Topic sections with key points (attributed to speakers where possible)
3. Important quotes (with speaker attribution)
4. Action items or takeaways
5. References or resources mentioned

Use clear formatting with headings and bullet points. Output as plain text.""",

            "blog": f"""Convert the following podcast transcript into a blog post/article.

Podcast Title: {title}
{speaker_info}

Transcript (key excerpts with speaker labels):
{segment_text}

Write a well-structured blog post that:
1. Has an engaging introduction
2. Organizes content into logical sections with headings
3. Includes key quotes and insights
4. Has a conclusion with takeaways
5. Is readable and engaging (not just transcript dump)

Target length: 800-1200 words. Use markdown formatting. Output as plain text.""",
        }

        return prompts[mode]

    def _build_structured_prompt(self, transcript: dict, mode: AnalysisMode) -> str:
        """Build prompt requesting JSON structured output."""
        base_prompt = self._build_prompt(transcript, mode)
        return base_prompt + """\n\nIMPORTANT: Output your response as valid JSON with this structure:
{
  "mode": "summary|insights|notes|blog",
  "title": "Podcast title",
  "content": "Your analysis text",
  "topics": ["topic1", "topic2", ...],
  "key_entities": ["person1", "place1", "org1", ...],
  "key_points": ["point1", "point2", ...],
  "sentiment": "positive|neutral|negative",
  "insights_count": number_of_insights,
  "main_themes": ["theme1", "theme2", ...],
  "analysis_quality": 0.0-1.0
"""

    def analyze(
        self,
        transcript: dict,
        mode: AnalysisMode = "summary",
        use_structured: Optional[bool] = None,
        base_data_dir: str = "data",
    ) -> LLMAnalysisResult:
        """Analyze a transcript using the configured LLM."""
        use_structured = use_structured if use_structured is not None else self.config.enable_structured_output

        start_time = time.time()
        transcript_id = transcript.get("video_title", "unknown")
        video_id = transcript.get("video_id", "") if "video_id" in transcript else ""

        # Check cache — skip re-analyzing if we already have this (video_id, mode) combo
        if video_id and (video_id, mode) in self._cache:
            cached = self._cache[(video_id, mode)]
            logger.info(f"Cache hit: {video_id} mode={mode}, returning cached result ({cached.processing_time_seconds:.2f}s original)")
            return cached

        # Build prompt
        if use_structured:
            prompt = self._build_structured_prompt(transcript, mode)
        else:
            prompt = self._build_prompt(transcript, mode)

        # Build API request
        api_url = self._get_api_url()
        # Use mode-specific token limit
        mode_max_tokens = self.config.max_tokens_by_mode.get(mode, self.config.max_tokens)

        if self.config.provider == "ollama":
            request_payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": self.config.streaming,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": mode_max_tokens,
                },
            }
        elif self.config.provider == "lmstudio":
            request_payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": self.config.streaming,
                "max_tokens": mode_max_tokens,
                "temperature": self.config.temperature,
            }

        logger.info(f"Sending analysis request to {api_url} (mode={mode}, model={self.config.model}, max_tokens={mode_max_tokens})")

        try:
            response = self.client.post(api_url, json=request_payload)
            response.raise_for_status()

            if self.config.streaming:
                # Parse streaming response
                raw_response = ""
                if self.config.provider == "ollama":
                    # Ollama streaming: each line has {"response": "...
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    raw_response += chunk["response"]
                            except json.JSONDecodeError:
                                continue
                elif self.config.provider == "lmstudio":
                    # LM Studio completions streaming: SSE with "data: {" prefix, uses choices[0].text (NOT delta.content)
                    for line in response.iter_lines():
                        if line:
                            # iter_lines() may return bytes or str depending on requests version
                            if isinstance(line, bytes):
                                text = line.decode()
                            else:
                                text = line
                            # Strip SSE "data: " prefix
                            if text.startswith("data: "):
                                text = text[6:]
                            # Skip "[DONE]" marker
                            if text.strip() == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(text)
                                choices = chunk.get("choices", [])
                                if choices:
                                    # LM Studio completions uses "text" directly, not "delta.content"
                                    if "text" in choices[0]:
                                        raw_response += choices[0]["text"]
                                    elif "delta" in choices[0] and "content" in choices[0]["delta"]:
                                        raw_response += choices[0]["delta"]["content"]
                            except json.JSONDecodeError:
                                continue
            else:
                # Non-streaming response
                data = response.json()
                if self.config.provider == "ollama":
                    raw_response = data.get("response", "")
                elif self.config.provider == "lmstudio":
                    # LM Studio completions: uses choices[0].text (NOT message.content)
                    raw_response = data.get("choices", [{}])[0].get("text", "")

            # Parse structured output if requested
            structured_output = None
            if use_structured and raw_response:
                try:
                    # Strip markdown code fences and surrounding whitespace
                    cleaned = raw_response.strip()
                    cleaned = re.sub(r'^\x60\x60\x60(?:json)?\s*', '', cleaned)
                    cleaned = re.sub(r'\s*\x60\x60\x60$', '', cleaned)
                    cleaned = cleaned.strip()
                    # Try to extract JSON from cleaned response
                    json_start = cleaned.find("{")
                    # Find the LAST } after the first { (handles stray } at start)
                    json_end = cleaned.rfind("}", json_start)
                    if json_end == -1:
                        # Response truncated — use end of string as fallback
                        json_end = len(cleaned) - 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = cleaned[json_start:json_end + 1]
                        structured_output = json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    structured_output = None

            # Extract summary text
            summary_text = raw_response if not structured_output else structured_output.get("content", raw_response)

            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time

            # Save to file in per-video folder
            output_path = None
            if self.config.enable_structured_output:
                video_folder = os.path.join(
                    base_data_dir,
                    folder_manager.generate_output_folder_name(transcript_id),
                )
                os.makedirs(video_folder, exist_ok=True)

                analysis_folder = os.path.join(video_folder, "analysis")
                os.makedirs(analysis_folder, exist_ok=True)

                sanitized_title = folder_manager.sanitize_filename(transcript_id)
                output_path = os.path.join(
                    analysis_folder,
                    f"{sanitized_title}_{mode}_analysis.json",
                )
                analysis_data = {
                    "mode": mode,
                    "model": self.config.model,
                    "provider": self.config.provider,
                    "structured_output": structured_output,
                    "summary_text": summary_text,
                    "raw_response": raw_response,
                    "processing_time": processing_time,
                    "transcript_id": transcript_id,
                    "timestamp": str(__import__("datetime").datetime.now()),
                }
                with open(output_path, "w") as f:
                    json.dump(analysis_data, f, indent=2)
                logger.info(f"Analysis saved to: {output_path}")

                # Save markdown file for all modes
                clean_text = summary_text
                # Strip thinking tags and model artifacts
                thinking_pattern = re.compile(r'<thinking>.*?</thinking>', re.DOTALL)
                clean_text = thinking_pattern.sub('', clean_text)
                # Strip stray JSON brackets/code block markers at start/end
                clean_text = re.sub(r'^\s*\}\s*\n\s*```', '', clean_text)
                clean_text = re.sub(r'```$', '', clean_text)
                clean_text = clean_text.strip()

                md_path = os.path.join(
                    analysis_folder,
                    f"{sanitized_title}_{mode}.md",
                )
                with open(md_path, "w") as f:
                    f.write(f"# {transcript_id}\n\n")
                    f.write(f"{clean_text}\n")
                logger.info(f"Markdown {mode} saved to: {md_path}")

            result = LLMAnalysisResult(
                analysis_mode=mode,
                llm_model=self.config.model,
                provider=self.config.provider,
                summary_text=summary_text,
                structured_output=structured_output,
                raw_response=raw_response,
                processing_time_seconds=processing_time,
                transcript_id=transcript_id,
                output_path=output_path,
            )

            logger.info(f"Analysis complete: mode={mode}, time={processing_time:.2f}s")

            # Cache result for future reuse (only if video_id is known)
            if video_id:
                self._evict_cache()
                self._cache[(video_id, mode)] = result
                logger.debug(f"Cache stored: {video_id} mode={mode}")

            return result

        except httpx.ConnectError:
            logger.error(f"LLM connection failed: {api_url} is not reachable")
            # Don't cache error results
            return LLMAnalysisResult(
                analysis_mode=mode,
                llm_model=self.config.model,
                provider=self.config.provider,
                summary_text="ERROR: LLM service unavailable",
                structured_output=None,
                raw_response="",
                processing_time_seconds=0,
                transcript_id=transcript_id,
                output_path=None,
            )
        except httpx.TimeoutException:
            logger.error(f"LLM request timed out after {self.config.timeout_seconds}s")
            return LLMAnalysisResult(
                analysis_mode=mode,
                llm_model=self.config.model,
                provider=self.config.provider,
                summary_text="ERROR: LLM request timed out",
                structured_output=None,
                raw_response="",
                processing_time_seconds=0,
                transcript_id=transcript_id,
                output_path=None,
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM HTTP error: {e.response.status_code}")
            return LLMAnalysisResult(
                analysis_mode=mode,
                llm_model=self.config.model,
                provider=self.config.provider,
                summary_text=f"ERROR: HTTP {e.response.status_code}",
                structured_output=None,
                raw_response="",
                processing_time_seconds=0,
                transcript_id=transcript_id,
                output_path=None,
            )

    def check_availability(self) -> bool:
        """Check if the LLM service is available."""
        try:
            if self.config.provider == "ollama":
                response = self.client.get(f"{self.config.base_url}/api/tags")
                return response.status_code == 200
            elif self.config.provider == "lmstudio":
                response = self.client.get(f"{self.config.lmstudio_url}/v1/models")
                return response.status_code == 200
        except httpx.ConnectError:
            return False
        except Exception:
            return False

    def list_available_models(self) -> list[str]:
        """List available models on the LLM service."""
        try:
            if self.config.provider == "ollama":
                response = self.client.get(f"{self.config.base_url}/api/tags")
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            elif self.config.provider == "lmstudio":
                response = self.client.get(f"{self.config.lmstudio_url}/v1/models")
                data = response.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def warmup(self) -> bool:
        """Send a short test prompt to warm up the model before batch analysis.

        This reduces the first-call latency in a batch workflow, since
        the model is already loaded and ready.
        """
        test_prompt = "Summarize in one sentence: the importance of efficient data processing."
        api_url = self._get_api_url()

        payload = {}
        result = ""

        if self.config.provider == "ollama":
            payload = {
                "model": self.config.model,
                "prompt": test_prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 64},
            }
        elif self.config.provider == "lmstudio":
            payload = {
                "model": self.config.model,
                "prompt": test_prompt,
                "stream": False,
                "max_tokens": 64,
                "temperature": 0.3,
            }

        try:
            response = self.client.post(api_url, json=payload)
            response.raise_for_status()
            if self.config.provider == "ollama":
                result = response.json().get("response", "")
            elif self.config.provider == "lmstudio":
                result = response.json().get("choices", [{}])[0].get("text", "")
            logger.info(f"Warmup successful: '{result[:80]}...'")
            return True
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
            return False

    def batch_analyze(
        self,
        transcripts: list[dict],
        mode: AnalysisMode = "summary",
        base_data_dir: str = "data",
        warmup_first: bool = True,
    ) -> list[LLMAnalysisResult]:
        """Analyze multiple transcripts in sequence, reusing the same client/model.

        Benefits over individual analyze() calls:
        - Model stays loaded (no warmup per video)
        - Single HTTP client session
        - Cache accumulates across the batch
        - Progress logging per item

        Returns a list of results, one per transcript.
        """
        results = []

        if warmup_first:
            self.warmup()

        logger.info(f"Batch analysis started: {len(transcripts)} transcripts, mode={mode}")

        for i, transcript in enumerate(transcripts):
            video_id = transcript.get("video_id", "")
            title = transcript.get("video_title", f"transcript_{i}")
            logger.info(f"Batch item {i+1}/{len(transcripts)}: {title} (video_id={video_id})")

            result = self.analyze(transcript, mode=mode, base_data_dir=base_data_dir)
            results.append(result)

            # Quick status
            if result.summary_text.startswith("ERROR"):
                logger.warning(f"Batch item {i+1} failed: {result.summary_text[:60]}")
            else:
                logger.info(f"Batch item {i+1} done: {result.processing_time_seconds:.2f}s")

        logger.info(f"Batch analysis complete: {len(results)} results, cache size={len(self._cache)}")
        return results

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._cache_max_size,
            "keys": list(self._cache.keys()),
        }
