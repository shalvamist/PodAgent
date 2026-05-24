"""Whisper transcription module — speech-to-text with context-enhanced prompting."""

import whisper
import torch
import os
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Whisper model sizes ordered by size (small to large)
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
# VRAM requirements per model (approximate)
MODEL_VRAM_GB = {
    "tiny": 1,
    "base": 1,
    "small": 2,
    "medium": 4,
    "large-v3": 10,
    "large-v3-turbo": 6,
}


@dataclass
class TranscriptionResult:
    """Result of a Whisper transcription."""
    audio_path: str
    text: str
    language: str
    segments: list
    duration: float
    success: bool
    error: Optional[str]


class WhisperTranscriber:
    """Transcribe audio using OpenAI Whisper model with context-enhanced prompting."""

    def __init__(
        self,
        model: str = "turbo",
        language: Optional[str] = None,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        carry_initial_prompt: bool = False,
    ):
        self.model_name = model
        self.language = language
        self.beam_size = beam_size
        self._model = None
        self.initial_prompt = initial_prompt
        self.carry_initial_prompt = carry_initial_prompt

    def _resolve_model_name(self) -> str:
        """Resolve the model name, applying GPU VRAM-aware fallback."""
        model = self.model_name

        # Map common aliases
        if model == "turbo":
            model = "large-v3-turbo"
        elif model == "large":
            model = "large-v3"

        # Check GPU VRAM and apply fallback if needed
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            required_vram = MODEL_VRAM_GB.get(model, 10)

            if gpu_mem < required_vram:
                # Find the largest model that fits
                fallback = None
                for m in WHISPER_MODELS:
                    if MODEL_VRAM_GB.get(m, 10) <= gpu_mem:
                        fallback = m
                if fallback and fallback != model:
                    logger.warning(
                        f"GPU has {gpu_mem:.1f}GB VRAM, model {model} requires "
                        f"{required_vram}GB — falling back to {fallback}"
                    )
                    model = fallback

        logger.info(f"Using Whisper model: {model}")
        return model

    def _load_model(self):
        """Lazy-load the Whisper model with GPU detection."""
        if self._model is None:
            resolved_model = self._resolve_model_name()
            logger.info(f"Loading Whisper model: {resolved_model}")
            self._model = whisper.load_model(resolved_model)

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU detected: {gpu_mem:.1f}GB VRAM — model on CUDA")
                self._model.to("cuda")
            else:
                logger.info("No GPU available — running on CPU (fp32 mode)")
                logger.info("CPU inference will be ~2-3x slower than GPU")

        return self._model

    def build_context_prompt(self, video_info: dict) -> Optional[str]:
        """Build a context prompt from YouTube metadata for Whisper."""
        parts = []

        # Extract guest names from description using multiple patterns
        if "description" in video_info and video_info["description"]:
            desc = video_info["description"]
            guest_patterns = [
                # "with John Doe", "featuring Jane Smith"
                r'(?:with|featuring|guest|guests?|special guest|joined by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                # "John Doe discusses", "Jane Smith appears"
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:is\s+|joined\s+|appears\s+|discusses|talks\s+about|explains)',
            ]
            for pattern in guest_patterns:
                matches = re.findall(pattern, desc, re.IGNORECASE)
                parts.extend([name.strip() for name in matches])

        # Channel topic
        if "channel" in video_info and video_info["channel"]:
            parts.append(f"Channel: {video_info['channel']}")

        # Tags
        if "tags" in video_info and video_info["tags"]:
            parts.extend(video_info["tags"])

        # Title keywords (first 10 words)
        if "title" in video_info and video_info["title"]:
            title_words = video_info["title"].split()[:10]
            parts.extend(title_words)

        # Uploader name
        if "uploader" in video_info and video_info["uploader"]:
            parts.append(f"Host: {video_info['uploader']}")

        return " ".join(parts) if parts else None

    def transcribe(
        self,
        audio_path: str,
        video_info: Optional[dict] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file using Whisper, optionally with context prompt."""
        if not os.path.exists(audio_path):
            return TranscriptionResult(
                audio_path=audio_path,
                text="",
                language="",
                segments=[],
                duration=0,
                success=False,
                error=f"File not found: {audio_path}",
            )

        model = self._load_model()

        # Build context prompt from video metadata if provided
        prompt = self.initial_prompt
        if video_info and not prompt:
            prompt = self.build_context_prompt(video_info)

        if prompt:
            logger.info(f"Context prompt: {prompt[:200]}...")

        logger.info(f"Transcribing: {audio_path}")

        try:
            result = model.transcribe(
                audio_path,
                language=None if self.language == "auto" else self.language,
                beam_size=self.beam_size,
                initial_prompt=prompt,
                carry_initial_prompt=self.carry_initial_prompt,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(
                audio_path=audio_path,
                text="",
                language="",
                segments=[],
                duration=0,
                success=False,
                error=str(e),
            )

        text = result.get("text", "")
        language = result.get("language", "unknown")
        segments = result.get("segments", [])
        # Compute duration from last segment end time (Whisper doesn't return duration directly)
        duration = 0
        if segments:
            duration = segments[-1].get("end", 0)

        logger.info(f"Transcribed: {len(text)} chars, language={language}, {len(segments)} segments")
        return TranscriptionResult(
            audio_path=audio_path,
            text=text,
            language=language,
            segments=segments,
            duration=duration,
            success=True,
            error=None,
        )
