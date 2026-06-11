"""Speaker diarization module — pyannote.audio for speaker identification."""

import torch
import os
import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

from src.utils import retry


logger = logging.getLogger(__name__)

@dataclass
class DiarizationResult:
    """Result of speaker diarization."""
    audio_path: str
    speaker_segments: list
    num_speakers: int
    duration: float
    success: bool
    error: Optional[str]
    quality: Optional[float]  # Diarization quality metric (0-1)


class SpeakerDiarizer:
    """Speaker diarization using pyannote.audio."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        pipeline_name: str = "pyannote/speaker-diarization-community-1",
    ):
        self.hf_token = hf_token
        self.pipeline_name = pipeline_name
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the pyannote diarization pipeline."""
        if self._pipeline is None:
            logger.info(f"Loading diarization pipeline: {self.pipeline_name}")
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(
                self.pipeline_name,
                token=self.hf_token,
            )

            if torch.cuda.is_available():
                logger.info("GPU detected — sending pipeline to CUDA")
                self._pipeline.to(torch.device("cuda"))
            else:
                logger.info("No GPU available — running diarization on CPU")

        return self._pipeline

    def _convert_to_wav(self, audio_path: str) -> str:
        """Convert audio file to WAV format (pyannote requires WAV)."""
        if audio_path.endswith(".wav"):
            return audio_path

        # Create temp WAV file
        wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-ac", "1",  # mono
            "-ar", "16000",  # 16kHz sample rate
            "-y",
            wav_path,
        ]

        logger.info(f"Converting {audio_path} to WAV for diarization")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error(f"Audio conversion failed: {result.stderr.strip()}")
            return ""

        return wav_path

    @retry(max_attempts=3, delay=2)
    def diarize(self, audio_path: str) -> DiarizationResult:
        """Diarize an audio file to identify speakers."""
        if not os.path.exists(audio_path):
            return DiarizationResult(
                audio_path=audio_path,
                speaker_segments=[],
                num_speakers=0,
                duration=0,
                success=False,
                error=f"File not found: {audio_path}",
                quality=None,
            )

        pipeline = self._load_pipeline()

        # Convert to WAV if needed (may create a temp file)
        wav_path = self._convert_to_wav(audio_path)
        was_temp = wav_path != audio_path and os.path.basename(wav_path) != os.path.basename(audio_path)
        if not wav_path or not os.path.exists(wav_path):
            return DiarizationResult(
                audio_path=audio_path,
                speaker_segments=[],
                num_speakers=0,
                duration=0,
                success=False,
                error="Audio conversion to WAV failed",
                quality=None,
            )

        logger.info(f"Diarizing: {wav_path}")

        try:
            output = pipeline(wav_path)
            diarization = output.speaker_diarization

            speaker_segments = []
            for turn, speaker in diarization:
                speaker_segments.append({
                    "start": round(turn.start, 1),
                    "end": round(turn.end, 1),
                    "speaker": speaker,
                })

            num_speakers = len(set(s["speaker"] for s in speaker_segments))
            duration = diarization.duration if hasattr(diarization, "duration") else 0

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return DiarizationResult(
                audio_path=audio_path,
                speaker_segments=[],
                num_speakers=0,
                duration=0,
                success=False,
                error=str(e),
                quality=None,
            )

        # Calculate diarization quality metric (heuristic only — not ground-truth accurate)
        quality = None
        if speaker_segments:
            segment_count = len(speaker_segments)
            num_speakers = len(set(s["speaker"] for s in speaker_segments))

            # Heuristic: very short average segments (< 2s) suggest over-segmentation
            total_duration = sum(seg["end"] - seg["start"] for seg in speaker_segments)
            avg_segment_len = total_duration / segment_count if segment_count > 0 else float('inf')

            if avg_segment_len < 1.5:
                quality = 0.3  # Over-segmented — poor diarization
            elif num_speakers == 1 and segment_count > 20:
                quality = 0.4  # Single speaker but many segments — likely over-split
            else:
                quality = 0.7 + min(0.3, (num_speakers / 5))  # Reasonable baseline

        logger.info(f"Diarized: {num_speakers} speakers, {len(speaker_segments)} segments, quality={quality}")

        # Clean up temp WAV file if we created one
        if was_temp and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass  # Best effort cleanup

        return DiarizationResult(
            audio_path=audio_path,
            speaker_segments=speaker_segments,
            num_speakers=num_speakers,
            duration=duration,
            success=True,
            error=None,
            quality=quality,
        )
