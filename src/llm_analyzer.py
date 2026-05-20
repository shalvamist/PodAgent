"""LLM analyzer module — feeds structured transcripts to local LLMs (Ollama/LM Studio)."""

import json
import os
import logging
import httpx
from dataclasses import dataclass
from typing import Optional, Literal

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


class LLMAnalyzer:
    """Analyze podcast transcripts using local LLMs (Ollama or LM Studio)."""

    def __init__(self, config: Optional[LLMAnalyzerConfig] = None):
        self.config = config or LLMAnalyzerConfig()
        self.client = httpx.Client(
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        )
        logger.info(f"LLMAnalyzer initialized: provider={self.config.provider}, model={self.config.model}")

    def _get_api_url(self) -> str:
        """Get the correct API endpoint URL based on provider."""
        if self.config.provider == "ollama":
            return f"{self.config.base_url}/api/generate"
        elif self.config.provider == "lmstudio":
            return f"{self.config.lmstudio_url}/v1/completions"
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _build_prompt(self, transcript: dict, mode: AnalysisMode) -> str:
        """Build the prompt for LLM analysis based on mode and transcript content."""
        title = transcript.get("video_title", "Unknown")
        speakers = transcript.get("speakers", [])
        raw_text = transcript.get("raw_text", "")
        segments = transcript.get("segments", [])

        # Truncate raw text if too long (LLMs have context limits)
        max_text_length = 8000
        if len(raw_text) > max_text_length:
            # Use segments approach for long transcripts
            raw_text = raw_text[:max_text_length] + "\n\n[... transcript truncated ...]"

        speaker_info = ""
        if speakers:
            speaker_info = "\nSpeakers:\n"
            for sp in speakers:
                label = sp.get("label", "Unknown")
                speaker_info += f"  - {label}\n"

        prompts = {
            "summary": f"""Analyze the following podcast transcript and provide a concise summary.

Podcast Title: {title}
{speaker_info}

Transcript:
{raw_text}

Provide a 1-2 paragraph summary that captures:
1. The main topics discussed
2. Key points and arguments
3. Any conclusions or recommendations
4. Notable quotes or insights

Keep it concise and informative.""",

            "insights": f"""Extract key insights from the following podcast transcript.

Podcast Title: {title}
{speaker_info}

Transcript:
{raw_text}

Provide a structured list of key insights:
1. Main topics covered (with brief explanation)
2. Key arguments or positions taken
3. Notable data points or statistics mentioned
4. Action items or recommendations
5. Unexpected or surprising points

Format as bullet points with clear headings.""",

            "notes": f"""Generate structured notes from the following podcast transcript.

Podcast Title: {title}
{speaker_info}

Transcript:
{raw_text}

Create structured notes with:
1. Executive summary (2-3 sentences)
2. Topic sections with key points (attributed to speakers where possible)
3. Important quotes (with speaker attribution)
4. Action items or takeaways
5. References or resources mentioned

Use clear formatting with headings and bullet points.""",

            "blog": f"""Convert the following podcast transcript into a blog post/article.

Podcast Title: {title}
{speaker_info}

Transcript:
{raw_text}

Write a well-structured blog post that:
1. Has an engaging introduction
2. Organizes content into logical sections with headings
3. Includes key quotes and insights
4. Has a conclusion with takeaways
5. Is readable and engaging (not just transcript dump)

Target length: 800-1200 words. Use markdown formatting.""",
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
  "key_points": ["point1", "point2", ...],
  "duration_minutes": estimated_duration,
  "speakers_mentioned": ["speaker names"],
  "topics": ["topic1", "topic2", ...]
}"""

    def analyze(
        self,
        transcript: dict,
        mode: AnalysisMode = "summary",
        use_structured: Optional[bool] = None,
    ) -> LLMAnalysisResult:
        """Analyze a transcript using the configured LLM."""
        use_structured = use_structured if use_structured is not None else self.config.enable_structured_output

        start_time = os.times().user  # approximate start
        transcript_id = transcript.get("video_title", "unknown")

        # Build prompt
        if use_structured:
            prompt = self._build_structured_prompt(transcript, mode)
        else:
            prompt = self._build_prompt(transcript, mode)

        # Build API request
        api_url = self._get_api_url()

        if self.config.provider == "ollama":
            request_payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": self.config.streaming,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }
        elif self.config.provider == "lmstudio":
            request_payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": self.config.streaming,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }

        logger.info(f"Sending analysis request to {api_url} (mode={mode}, model={self.config.model})")

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
                    # Try to extract JSON from response
                    json_start = raw_response.find("{")
                    json_end = raw_response.rfind("}")
                    if json_start >= 0 and json_end > json_start:
                        json_str = raw_response[json_start:json_end + 1]
                        structured_output = json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    structured_output = None

            # Extract summary text
            summary_text = raw_response if not structured_output else structured_output.get("content", raw_response)

            # Calculate processing time
            end_time = os.times().user
            processing_time = end_time - start_time

            # Save to file
            output_path = None
            if self.config.enable_structured_output:
                output_dir = "data/llm_analysis"
                os.makedirs(output_dir, exist_ok=True)
                sanitized_title = self._sanitize_filename(transcript_id)
                output_path = os.path.join(
                    output_dir,
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
                
                # For summary mode, also save a markdown file
                if mode == "summary":
                    clean_text = summary_text
                    # Strip thinking tags if present (model may wrap output in tags)
                    import re
                    thinking_pattern = re.compile(r'<thinking>.*?</thinking>', re.DOTALL)
                    clean_text = thinking_pattern.sub('', clean_text)
                    clean_text = clean_text.strip()
                    
                    md_path = os.path.join(
                        output_dir,
                        f"{sanitized_title}_summary.md",
                    )
                    with open(md_path, "w") as f:
                        f.write(f"# {transcript_id}\n\n")
                        f.write(f"{clean_text}\n")
                    logger.info(f"Markdown summary saved to: {md_path}")

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
            return result

        except httpx.ConnectError:
            logger.error(f"LLM connection failed: {api_url} is not reachable")
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

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename."""
        invalid_chars = '<>:"/\\|?*'
        for c in invalid_chars:
            name = name.replace(c, "_")
        return name[:100]

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
