"""PodAgent TTS Generator — convert summary text to speech audio."""

import asyncio
import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """Configuration for TTS generator."""
    provider: str = "edge-tts"  # "edge-tts" or "elevenlabs"
    voice: str = "en-US-AvaMultilingualNeural"  # Voice name
    rate: str = "+0%"  # Speech rate
    pitch: str = "+0Hz"  # Speech pitch
    output_format: str = "mp3"  # Output format
    elevenlabs_api_key: str = ""  # API key for ElevenLabs
    elevenlabs_model: str = "eleven_multilingual_v2"  # ElevenLabs model


class TTSGenerator:
    """Text-to-speech generator supporting edge-tts and ElevenLabs."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._provider = self.config.provider

    def generate(self, text: str, output_path: str) -> dict:
        """Generate TTS audio from text. Returns {success, output_path, duration, error}."""
        if self._provider == "elevenlabs":
            return self._generate_elevenlabs(text, output_path)
        else:
            return self._generate_edge_tts(text, output_path)

    def _generate_edge_tts(self, text: str, output_path: str) -> dict:
        """Generate audio using Microsoft Edge TTS (free, neural voices)."""
        try:
            import edge_tts

            # Clean text for TTS (remove markdown artifacts)
            clean_text = self._clean_text_for_tts(text)

            logger.info(f"Generating TTS via edge-tts: voice={self.config.voice}, rate={self.config.rate}")

            communicate = edge_tts.Communicate(clean_text, self.config.voice, rate=self.config.rate, pitch=self.config.pitch)
            asyncio.run(communicate.save(output_path))

            file_size = os.path.getsize(output_path)
            logger.info(f"TTS audio saved: {output_path} ({file_size / 1024:.1f}KB)")

            return {"success": True, "output_path": output_path, "file_size": file_size, "error": None}

        except Exception as e:
            logger.error(f"edge-tts generation failed: {e}")
            return {"success": False, "output_path": None, "file_size": 0, "error": str(e)}

    def _generate_elevenlabs(self, text: str, output_path: str) -> dict:
        """Generate audio using ElevenLabs REST API (premium, highest quality)."""
        try:
            import requests

            if not self.config.elevenlabs_api_key:
                logger.warning("ElevenLabs API key not set — falling back to edge-tts")
                return self._generate_edge_tts(text, output_path)

            clean_text = self._clean_text_for_tts(text)

            logger.info(f"Generating TTS via ElevenLabs: voice={self.config.voice}, model={self.config.elevenlabs_model}")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.config.voice}",
                headers={
                    "xi-api-key": self.config.elevenlabs_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": clean_text,
                    "model_id": self.config.elevenlabs_model,
                    "output_format": self.config.output_format,
                },
                stream=True,
            )

            if response.status_code != 200:
                error_msg = response.json().get("detail", response.text)
                logger.error(f"ElevenLabs API error: {error_msg}")
                return {"success": False, "output_path": None, "file_size": 0, "error": error_msg}

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(output_path)
            logger.info(f"TTS audio saved: {output_path} ({file_size / 1024:.1f}KB)")

            return {"success": True, "output_path": output_path, "file_size": file_size, "error": None}

        except Exception as e:
            logger.error(f"ElevenLabs generation failed: {e}")
            return {"success": False, "output_path": None, "file_size": 0, "error": str(e)}

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS — remove markdown, thinking tags, JSON artifacts."""
        import re

        # Strip thinking tags
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        # Strip code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Strip JSON brackets at start/end
        text = re.sub(r'^\s*\}\s*\n\s*```', '', text)
        text = re.sub(r'```$', '', text)
        # Remove markdown headers
        text = re.sub(r'^#\s+.*?\n', '', text)
        # Remove bold markers
        text = re.sub(r'\*\*', '', text)
        # Remove bullet markers
        text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def list_available_voices(self) -> list:
        """List available TTS voices for the configured provider."""
        if self._provider == "elevenlabs":
            return self._list_elevenlabs_voices()
        else:
            return self._list_edge_tts_voices()

    def _list_edge_tts_voices(self) -> list:
        """List available Edge TTS voices."""
        try:
            import edge_tts
            voices = asyncio.run(edge_tts.list_voices())
            # Filter English neural voices
            en_voices = [v for v in voices if v['Locale'].startswith('en-')]
            logger.info(f"Edge TTS: {len(en_voices)} English voices available")
            return en_voices[:20]  # Return top 20
        except Exception as e:
            logger.error(f"Failed to list edge-tts voices: {e}")
            return []

    def _list_elevenlabs_voices(self) -> list:
        """List available ElevenLabs voices."""
        try:
            import elevenlabs
            if not self.config.elevenlabs_api_key:
                logger.warning("ElevenLabs API key not set")
                return []
            voices = elevenlabs.voices()
            logger.info(f"ElevenLabs: {len(voices)} voices available")
            return voices[:20]
        except Exception as e:
            logger.error(f"Failed to list elevenlabs voices: {e}")
            return []
