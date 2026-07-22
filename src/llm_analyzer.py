"""LLM analyzer module — feeds structured transcripts to local LLMs (Ollama/LM Studio)."""

import json
import os
import time
import logging
import httpx
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Literal

from src import folder_manager

logger = logging.getLogger(__name__)

# Analysis mode types
AnalysisMode = Literal["summary", "insights", "notes", "blog"]

# Qwen/DeepSeek thinking tag — different from XML-style <thinking>
_QWEN_THINK_OPEN = chr(0x3C) + chr(0x74) + chr(0x68) + chr(0x69) + chr(0x6E) + chr(0x6B) + chr(0x3E)  # <thinking>
_QWEN_THINK_CLOSE = chr(0x3C) + chr(0x2F) + chr(0x74) + chr(0x68) + chr(0x69) + chr(0x6E) + chr(0x6B) + chr(0x3E)  # </thinking>


def _extract_json(text: str) -> dict:
    """Extract valid JSON from LLM response, handling thinking blocks and stray chars.

    Many models (Qwen, DeepSeek, etc.) output reasoning text before/around the JSON.
    This function tries multiple strategies to find parseable JSON.
    """
    if not text:
        raise ValueError("Empty input")

    # Strategy 1: Strip code fences and thinking blocks, then find first { ... last }
    cleaned = text.strip()
    cleaned = re.sub(r'^\x60\x60\x60(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*\x60\x60\x60$', '', cleaned).strip()

    # Remove thinking blocks: both <thinking> (XML-style) and  (Qwen/DeepSeek style)
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(re.escape(_QWEN_THINK_OPEN) + '.*?' + re.escape(_QWEN_THINK_CLOSE), '', cleaned, flags=re.DOTALL)

    json_start = cleaned.find("{")
    if json_start >= 0:
        # Find the LAST } after the first { (handles stray } at start, nested objects)
        json_end = cleaned.rfind("}", json_start)
        if json_end > json_start:
            candidate = cleaned[json_start:json_end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Strategy 2: Strip thinking blocks first, then try JSON extraction.
    # Thinking blocks start with markers like "Here's a thinking process:", "<thinking>", etc.
    # and end at the first real content heading or JSON object.
    cleaned = re.sub(
        r'(?:^|\n)\s*(?:```(?:json)?\s*\n)?'
        r'[{\[]?\s*'
        r'(?:<thinking>|Here\'?s a thinking process|Thinking process|'
         'Let me think|Step-by-step)'
        r'.*?'
        r'(?=\{)',  # stop at first { after thinking block
        '', text, flags=re.DOTALL | re.MULTILINE,
    )

    json_start = cleaned.find("{")
    if json_start >= 0:
        json_end = cleaned.rfind("}", json_start)
        if json_end > json_start:
            candidate = cleaned[json_start:json_end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Strategy 3: Find the largest valid JSON object by scanning all { ... } pairs.
    # This handles cases where thinking text is interleaved with JSON fragments.
    best_json = None
    best_len = 0
    for i, ch in enumerate(cleaned):
        if ch == '{':
            j = cleaned.rfind("}", i)
            if j > i:
                candidate = cleaned[i:j + 1]
                # Skip tiny fragments (likely stray braces in thinking text)
                if len(candidate) < 50:
                    continue
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and len(candidate) > best_len:
                        best_json = obj
                        best_len = len(candidate)
                except json.JSONDecodeError:
                    pass

    if best_json is not None:
        return best_json

    raise ValueError("No valid JSON found in response")


def _strip_thinking_blocks(text: str) -> str:
    """Remove XML-style and Qwen/DeepSeek thinking tags from LLM output."""
    cleaned = text.strip()

    # Remove paired thinking blocks (both XML-style and Qwen/DeepSeek style)
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(
        re.escape(_QWEN_THINK_OPEN) + '.*?' + re.escape(_QWEN_THINK_CLOSE),
        '', cleaned, flags=re.DOTALL,
    )

    # Handle unclosed Qwen/DeepSeek thinking tags — strip up to first real content
    if _QWEN_THINK_OPEN in cleaned:
        idx = cleaned.find(_QWEN_THINK_OPEN)
        after_tag = cleaned[idx:]
        json_match = re.search(r'\{', after_tag)
        if json_match and json_match.start() > 0:
            cleaned = cleaned[:idx] + after_tag[json_match.start():]
        else:
            # Try various content markers that indicate real output has started
            content_patterns = [
                r'(?<=\n)\s*#{1,6}\s',           # Markdown heading (### Heading)
                r'\*\*[^\n]+\*\*',               # Bold text (**text**)
                r'^[A-Z][a-z]+:',                # Capitalized label at line start
            ]
            content_match = None
            for pat in content_patterns:
                m = re.search(pat, after_tag)
                if m and m.start() > 0:
                    content_match = m
                    break
            if content_match:
                cleaned = cleaned[:idx] + after_tag[content_match.start():]
            else:
                # No content marker found — entire response is thinking text.
                # Return empty to avoid leaking raw thinking into output.
                cleaned = ""

    # Handle unclosed XML-style <thinking> tags (model sometimes omits </thinking>)
    if '<thinking>' in cleaned and '</thinking>' not in cleaned:
        idx = cleaned.find('<thinking>')
        after_tag = cleaned[idx:]
        json_match = re.search(r'\{', after_tag)
        if json_match and json_match.start() > 0:
            cleaned = cleaned[:idx] + after_tag[json_match.start():]
        else:
            content_patterns = [
                r'(?<=\n)\s*#{1,6}\s',
                r'\*\*[^\n]+\*\*',
                r'^[A-Z][a-z]+:',
            ]
            content_match = None
            for pat in content_patterns:
                m = re.search(pat, after_tag)
                if m and m.start() > 0:
                    content_match = m
                    break
            if content_match:
                cleaned = cleaned[:idx] + after_tag[content_match.start():]
            else:
                cleaned = ""

    return cleaned


def _strip_stray_chars(text: str) -> str:
    """Remove leading stray characters (e.g. '}' from JSON bleed) and trailing code fences."""
    cleaned = text.strip()
    # Strip leading stray braces/brackets on their own line
    cleaned = re.sub(r'^\s*[}\]\)]+\n\s*\n?', '', cleaned)
    # Remove trailing code fence markers
    cleaned = re.sub(r'\s*```(?:json)?\s*$', '', cleaned)
    return cleaned


def _strip_planning_text(text: str) -> str:
    """Remove draft structure / planning text that precedes actual content."""
    cleaned = text.strip()

    # Remove "Draft structure" / planning text preceding numbered headings
    cleaned = re.sub(
        r'(?:^|\n)\s*(?:Draft\s+structure|Plan|Outline).*?(?=\*\*[\d\.]+\.\s)',
        '', cleaned, flags=re.DOTALL | re.MULTILINE,
    )

    # Remove trailing planning/checking text (e.g., "Check against JSON schema:")
    cleaned = re.sub(
        r'(?:^|\n)\s*(?:Check\s+against|Verify|Validate).*$',
        '', cleaned, flags=re.DOTALL | re.MULTILINE,
    )

    # Remove plain-text thinking process blocks — common with Qwen, DeepSeek, etc.
    thinking_block = re.compile(
        r'(?:^|\n)\s*(?:```(?:json)?\s*\n)?'
        r'[{\\[]?\s*'
        r'(?:<thinking>|Here\'?s a thinking process|Thinking process|'
         'Let me think|Step-by-step)'
        r'.*?'
        r'(?=^\s{0,4}\*\*[\d\.]+\.\s)',  # stop at first bold numbered heading like **1.
        re.DOTALL | re.MULTILINE,
    )
    cleaned = thinking_block.sub('', cleaned)

    # Remove remaining plain-text thinking blocks with no real content after them
    if 'Thinking Process' in cleaned or "Here's a thinking process" in cleaned:
        think_match = re.search(
            r'(?:^|\n).*?(?:Thinking Process|Here\'?s a thinking process)',
            cleaned, re.DOTALL,
        )
        if think_match:
            after_think = cleaned[think_match.end():]
            has_content = bool(
                '{' in after_think or
                re.search(r'(?:^\s*\d+\.\s+.{30,}|\s*[-*]\s+[A-Z].{40})', after_think, re.MULTILINE)
            )
            if not has_content:
                cleaned = cleaned[:think_match.start()]

    # Remove any leftover thinking tag text that may have bled into content lines
    cleaned = re.sub(re.escape(_QWEN_THINK_OPEN) + r'\s*', '', cleaned)

    return cleaned


def _clean_plain_response(text: str) -> str:
    """Strip model artifacts from raw LLM response for plain-text output.

    Only strips thinking blocks and stray chars — no JSON template parsing.
    This avoids the Qwen confusion where it fills a JSON template with placeholders.
    """
    if not text:
        return ""
    cleaned = _strip_thinking_blocks(text)
    cleaned = _strip_stray_chars(cleaned)
    # Strip planning text like "Draft looks solid", "Proceeds" etc.
    cleaned = _strip_planning_text(cleaned)
    return cleaned.strip() if cleaned else ""


def _extract_json_content(text: str) -> Optional[str]:
    """Try to extract 'content' value from JSON objects in the text.

    Handles both valid and truncated/partial JSON. Returns None if no content found.
    Skips template placeholders (short, generic content).
    """
    cleaned = text.strip()
    if '{' not in cleaned:
        return None

    # Strategy 1: Find all brace positions and try to parse complete JSON objects
    candidates = []
    for i, c in enumerate(cleaned):
        if c == '{':
            depth = 0
            found_end = False
            for j in range(i, len(cleaned)):
                if cleaned[j] == '{':
                    depth += 1
                elif cleaned[j] == '}':
                    depth -= 1
                    if depth == 0:
                        candidates.append(cleaned[i:j+1])
                        found_end = True
                        break
            if not found_end:
                candidates.append(cleaned[i:])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and 'content' in obj:
                content_val = str(obj['content']).strip()
                if len(content_val) > 50 and 'Your analysis' not in content_val and 'topic1' not in content_val:
                    return content_val
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 2: Handle truncated JSON — extract "content" value manually
    if '{' in cleaned and '"content"' in cleaned:
        brace_positions = [i for i, c in enumerate(cleaned) if c == '{']
        for start_pos in reversed(brace_positions):
            block = cleaned[start_pos:]

            idx = block.find('"content"')
            if idx < 0:
                continue

            colon_idx = block.find(':', idx)
            if colon_idx < 0:
                continue

            start_quote = block.find('"', colon_idx + 1)
            if start_quote < 0:
                continue

            content_start = start_quote + 1
            remaining = block[content_start:]

            in_escape = False
            depth = 0
            end_pos = len(remaining) - 1

            for i, c in enumerate(remaining):
                if in_escape:
                    in_escape = False
                    continue
                if c == '\\':
                    in_escape = True
                    continue
                elif c == '"':
                    depth += 1
                    if depth % 2 == 0:
                        rest = remaining[i + 1:]
                        next_key = re.match(r'\s*,\s*\n\s{0,6}"(\w+)"\s*:', rest)
                        if next_key:
                            end_pos = i + 1
                            break

            raw_content = block[content_start:content_start + end_pos]
            fixed_content = (raw_content
                             .replace('\\"', '"')
                             .replace('\\\\n', '\n')
                             .replace('\\\\', '\\')
                             .strip())

            if len(fixed_content) > 50 and 'Your analysis' not in fixed_content and 'topic1' not in fixed_content:
                return fixed_content

    return None


def _extract_non_json_content(text: str) -> Optional[str]:
    """Extract non-JSON analysis text (bullet points, numbered sections).

    Returns cleaned content if real analysis is found, or None.
    """
    cleaned = text.strip()

    # Check for real analysis text after thinking (bullet points, numbered sections)
    has_real_analysis = bool(
        re.search(r'(?:^\s*\d+\.\s+.*(?:\n|$)|^\s*[-*]\s+[A-Z].{20})', cleaned, re.MULTILINE)
    )

    if has_real_analysis:
        # Keep the analysis text — strip only thinking markers and stray braces
        cleaned = re.sub(
            r'(?:^|\n).*?(?:Thinking Process|Here\'?s a thinking process)',
            '', cleaned, count=1, flags=re.DOTALL,
        )
        cleaned = re.sub(r'\{[^}]*"Your analysis text"[^}]*\}', '', cleaned, flags=re.DOTALL)
    else:
        # No real content — strip all thinking and template artifacts
        cleaned = re.sub(
            r'(?:^|\n).*?(?:Thinking Process|Here\'?s a thinking process)',
            '', cleaned, count=1, flags=re.DOTALL,
        )
        cleaned = re.sub(r'\n[^\n]*', '', cleaned, count=1)
        cleaned = re.sub(r'\{[^}]*"Your analysis text"[^}]*\}', '', cleaned, flags=re.DOTALL)
        # Strip numbered planning outlines
        cleaned = re.sub(
            r'^\s*\d+\.\s*\*\*.*?(?:Deconstruct|Outline|Structure|Plan).*?\*\*',
            '', cleaned, flags=re.MULTILINE,
        )
        # If remaining text looks like an outline (bullet points with section headers), strip it all
        if re.search(r'^-\s+\*[A-Z]', cleaned, re.MULTILINE):
            return None

    result = cleaned.strip()
    if result == ':' or not result:
        return None
    return result


def _clean_raw_response(text: str) -> str:
    """Strip model artifacts from raw LLM response when JSON parsing failed.

    Pipeline: strip thinking blocks → remove stray chars → remove planning text →
    try JSON extraction → fall back to non-JSON content extraction.
    """
    if not text:
        return ""

    cleaned = _strip_thinking_blocks(text)
    cleaned = _strip_stray_chars(cleaned)
    cleaned = _strip_planning_text(cleaned)

    # Try to extract JSON content first (preferred — clean structured output)
    json_content = _extract_json_content(cleaned)
    if json_content is not None:
        return json_content

    # Fall back to non-JSON analysis text
    non_json = _extract_non_json_content(cleaned)
    if non_json is not None:
        return non_json

    return ""


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
    # Mode-specific token limits (summary/insights don't need full 4096)
    max_tokens_by_mode: dict = None  # Will be set in __init__


class LLMAnalyzer:
    """Analyze podcast transcripts using local LLMs (Ollama or LM Studio)."""

    def __init__(self, config: Optional[LLMAnalyzerConfig] = None):
        self.config = config or LLMAnalyzerConfig()
        # Token limits — generous enough for full analysis of 3h+ podcasts (80k context window models)
        self.config.max_tokens_by_mode = {
            "summary": 8192,     # Full summary with structured JSON
            "insights": 8192,    # Comprehensive insights with structured JSON
            "notes": 8192,       # Detailed notes with structured JSON
            "blog": 16384,       # Long-form blog post with structured JSON
        }
        self.client = httpx.Client(
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        )
        # Result cache: keyed by (video_id, mode, chunk_index) — skip re-analyzing same combo
        self._cache: dict[tuple[str, str, Optional[int]], LLMAnalysisResult] = {}
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

    def _resolve_model(self) -> str:
        """Resolve the model name to use for this request.

        For Ollama, returns the configured model from config.yaml.
        For LM Studio, queries /v1/models and uses whatever is currently loaded — no hardcoded model needed.
        Falls back to the configured model if discovery fails.
        """
        if self.config.provider != "lmstudio":
            return self.config.model

        try:
            resp = self.client.get(f"{self.config.lmstudio_url}/v1/models")
            if resp.status_code == 200 and resp.json().get("data"):
                active_model = resp.json()["data"][0]["id"]
                if active_model != self.config.model:
                    logger.info(
                        f"LM Studio has model '{active_model}' loaded (config says '{self.config.model}' — using actual)"
                    )
                return active_model
        except Exception as e:
            logger.warning(f"Failed to query LM Studio /v1/models: {e} — falling back to config model")

        return self.config.model

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

    def analyze(
        self,
        transcript: dict,
        mode: AnalysisMode = "summary",
        base_data_dir: str = "data",
        video_folder: Optional[str] = None,
    ) -> LLMAnalysisResult:
        """Analyze a transcript using the configured LLM.

        Always uses plain-text prompts — no JSON template requests.
        The model produces clean markdown directly, which is written to file.
        """
        start_time = time.time()
        transcript_id = transcript.get("video_title", "unknown")
        video_id = transcript.get("video_id", "") if "video_id" in transcript else ""

        # Resolve the actual model name (for LM Studio, discovers whatever is loaded)
        resolved_model = self._resolve_model()

        # Check cache — skip re-analyzing if we already have this (video_id, mode, chunk_index) combo
        chunk_idx = transcript.get("chunk_index")
        cache_key = (video_id, mode, chunk_idx)
        if video_id and cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.info(f"Cache hit: {video_id} mode={mode}, returning cached result ({cached.processing_time_seconds:.2f}s original)")
            return cached

        # Pre-flight: verify LLM service is reachable before wasting time
        is_available, error_msg = self.check_availability()
        if not is_available:
            logger.error(f"LLM analysis aborted — {error_msg}")
            return LLMAnalysisResult(
                analysis_mode=mode,
                llm_model=resolved_model,
                provider=self.config.provider,
                summary_text=f"ERROR: {error_msg}",
                structured_output=None,
                raw_response="",
                processing_time_seconds=0,
                transcript_id=transcript_id,
                output_path=None,
            )

        # Build plain-text prompt (no JSON template — avoids model confusion)
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
                "model": resolved_model,
                "prompt": prompt,
                "stream": self.config.streaming,
                "max_tokens": mode_max_tokens,
                "temperature": self.config.temperature,
            }

        logger.info(f"Sending analysis request to {api_url} (mode={mode}, model={resolved_model}, max_tokens={mode_max_tokens})")

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

            # Clean model artifacts (thinking blocks, stray chars, planning text) — no JSON template parsing
            summary_text = _clean_plain_response(raw_response) if raw_response else ""

            # Retry once if Qwen produced only thinking text with no real content.
            # This happens when the model gets stuck in "thinking mode" and never
            # closes its thinking block or produces output after it (common for longer prompts).
            retry_count = 0
            while not summary_text and raw_response:
                if retry_count >= 1:
                    logger.warning(f"No content extracted from LLM response for mode={mode} after cleanup")
                    break
                retry_count += 1
                logger.info(f"Retrying LLM analysis (mode={mode}) — previous attempt produced only thinking text")

                # Retry with slightly lower temperature to reduce over-thinking
                old_temp = self.config.temperature
                self.config.temperature = max(0.3, old_temp - 0.2)
                try:
                    response = self.client.post(api_url, json=request_payload)
                    response.raise_for_status()

                    if self.config.streaming:
                        raw_response = ""
                        if self.config.provider == "ollama":
                            for line in response.iter_lines():
                                if line:
                                    try:
                                        chunk = json.loads(line)
                                        if "response" in chunk:
                                            raw_response += chunk["response"]
                                    except json.JSONDecodeError:
                                        continue
                        elif self.config.provider == "lmstudio":
                            for line in response.iter_lines():
                                if line:
                                    text = line.decode() if isinstance(line, bytes) else line
                                    if text.startswith("data: "):
                                        text = text[6:]
                                    if text.strip() == "[DONE]":
                                        continue
                                    try:
                                        chunk = json.loads(text)
                                        choices = chunk.get("choices", [])
                                        if choices:
                                            if "text" in choices[0]:
                                                raw_response += choices[0]["text"]
                                            elif "delta" in choices[0] and "content" in choices[0]["delta"]:
                                                raw_response += choices[0]["delta"]["content"]
                                    except json.JSONDecodeError:
                                        continue
                    else:
                        data = response.json()
                        if self.config.provider == "ollama":
                            raw_response = data.get("response", "")
                        elif self.config.provider == "lmstudio":
                            raw_response = data.get("choices", [{}])[0].get("text", "")

                    summary_text = _clean_plain_response(raw_response) if raw_response else ""
                finally:
                    self.config.temperature = old_temp

            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time

            # Save to file in per-video folder — reuse video_folder if provided (from download time), else generate one
            vf = video_folder or os.path.join(base_data_dir, folder_manager.generate_output_folder_name(transcript_id))
            os.makedirs(vf, exist_ok=True)

            analysis_folder = os.path.join(vf, "analysis")
            os.makedirs(analysis_folder, exist_ok=True)

            sanitized_title = folder_manager.sanitize_filename(transcript_id)
            chunk_suffix = f"_c{chunk_idx}" if chunk_idx is not None else ""
            output_path = os.path.join(
                analysis_folder,
                f"{sanitized_title}{chunk_suffix}_{mode}_analysis.json",
            )
            analysis_data = {
                "mode": mode,
                "model": resolved_model,
                "provider": self.config.provider,
                "summary_text": summary_text,
                "raw_response": raw_response,
                "processing_time": processing_time,
                "transcript_id": transcript_id,
                "timestamp": str(datetime.now()),
            }
            with open(output_path, "w") as f:
                json.dump(analysis_data, f, indent=2)
            logger.info(f"Analysis saved to: {output_path}")

            # Save markdown file — Qwen produces clean output directly (no JSON template confusion)
            md_path = os.path.join(
                analysis_folder,
                f"{sanitized_title}{chunk_suffix}_{mode}.md",
            )
            with open(md_path, "w") as f:
                f.write(f"# {transcript_id}\n\n")
                f.write(f"{summary_text}\n")
            logger.info(f"Markdown {mode} saved to: {md_path}")

            result = LLMAnalysisResult(
                analysis_mode=mode,
                llm_model=self.config.model,
                provider=self.config.provider,
                summary_text=summary_text,
                structured_output=None,  # No longer using JSON templates
                raw_response=raw_response,
                processing_time_seconds=processing_time,
                transcript_id=transcript_id,
                output_path=output_path,
            )

            logger.info(f"Analysis complete: mode={mode}, time={processing_time:.2f}s")

            # Cache result for future reuse (only if video_id is known)
            if video_id:
                cache_key = (video_id, mode, transcript.get("chunk_index"))
                self._evict_cache()
                self._cache[cache_key] = result
                logger.debug(f"Cache stored: {video_id} mode={mode} chunk={transcript.get('chunk_index')}")

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
            resp_text = ""
            try:
                resp_text = e.response.text[:300]
            except Exception:
                pass
            logger.error(f"LLM HTTP error: {e.response.status_code} — body: {resp_text}")
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

    def check_availability(self) -> tuple[bool, Optional[str]]:
        """Check if the LLM service is available and loaded.

        Returns:
            (is_available, error_message_or_none) — when unavailable, the message
            tells the user exactly what to do to fix it.
        """
        try:
            if self.config.provider == "ollama":
                response = self.client.get(f"{self.config.base_url}/api/tags")
                if response.status_code != 200:
                    return False, (
                        f"Ollama responded with HTTP {response.status_code}. "
                        "Make sure Ollama is running (`ollama serve`) and a model is loaded."
                    )
                return True, None

            elif self.config.provider == "lmstudio":
                response = self.client.get(f"{self.config.lmstudio_url}/v1/models")
                if response.status_code != 200:
                    return False, (
                        f"LM Studio responded with HTTP {response.status_code}. "
                        "Make sure LM Studio is running and a model is loaded in the UI."
                    )

                # If no models listed but we have a config default, that's OK — use it.
                data = response.json()
                models = data.get("data", [])
                if not models:
                    logger.warning(
                        "LM Studio has no model loaded in the UI — will use configured fallback model '%s'. "
                        "For best results, load a model manually in LM Studio.",
                        self.config.model,
                    )
                return True, None

        except httpx.ConnectError:
            if self.config.provider == "ollama":
                return False, (
                    f"Cannot connect to Ollama at {self.config.base_url}. "
                    "Start Ollama with `ollama serve` in a terminal before running analysis."
                )
            else:
                return False, (
                    f"Cannot connect to LM Studio at {self.config.lmstudio_url}. "
                    "1. Open the LM Studio app on your computer.\n"
                    "2. Load a model (download one from the search tab if needed).\n"
                    "3. Go to the 'Local Server' tab and start the server.\n"
                    "4. Run PodAgent again with --analyze."
                )

        except httpx.ConnectTimeout:
            if self.config.provider == "ollama":
                return False, (
                    f"Connection to Ollama at {self.config.base_url} timed out. "
                    "Make sure `ollama serve` is running and not blocked by a firewall."
                )
            else:
                return False, (
                    f"Connection to LM Studio at {self.config.lmstudio_url} timed out. "
                    "1. Make sure the LM Studio app is open.\n"
                    "2. A model must be loaded before starting the local server.\n"
                    "3. Go to 'Local Server' tab and start the server."
                )

        except httpx.ReadTimeout:
            if self.config.provider == "ollama":
                return False, (
                    f"Ollama at {self.config.base_url} did not respond in time. "
                    "A model may be loading — wait a moment and try again."
                )
            else:
                return False, (
                    f"LM Studio at {self.config.lmstudio_url} did not respond in time. "
                    "A model may be loading into memory — wait for it to finish loading "
                    "(check the LM Studio UI) and try again."
                )

        except Exception as e:
            provider_name = {"ollama": "Ollama", "lmstudio": "LM Studio"}.get(
                self.config.provider, self.config.provider.capitalize()
            )
            return False, (
                f"Unexpected error checking {provider_name}: {e}. "
                f"Make sure your {provider_name} server is running on the correct URL."
            )

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
            resolved = self._resolve_model()
            payload = {
                "model": resolved,
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
