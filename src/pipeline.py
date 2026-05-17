"""PodAgent pipeline — end-to-end podcast processing."""

import json
import os
import logging
import yaml
import re
from dataclasses import dataclass
from typing import Optional

from src.downloader import YouTubeAudioDownloader
from src.transcriber import WhisperTranscriber
from src.diarizer import SpeakerDiarizer
from src.llm_analyzer import LLMAnalyzer
from src.storage import PodcastStorage

logger = logging.getLogger(__name__)


class PodAgentPipeline:
    """End-to-end podcast processing pipeline."""

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            with open("config.yaml") as f:
                config = yaml.safe_load(f)
        self.config = config

        # Initialize components
        self.downloader = YouTubeAudioDownloader(config=self.config)
        self.transcriber = WhisperTranscriber(config=self.config)
        self.diarizer = SpeakerDiarizer(config=self.config)

        # LLM config
        self.llm_config = LLMAnalyzerConfig(
            provider=self.config["llm_analyzer"]["provider"],
            model=self.config["llm_analyzer"]["model"],
            lmstudio_url=self.config["llm_analyzer"]["lmstudio_url"],
            temperature=self.config["llm_analyzer"]["temperature"],
            max_tokens=self.config["llm_analyzer"]["max_tokens"],
            timeout_seconds=self.config["llm_analyzer"]["timeout_seconds"],
            streaming=self.config["llm_analyzer"]["streaming"],
            output_format=self.config["llm_analyzer"]["output_format"],
            enable_structured_output=self.config["llm_analyzer"]["enable_structured_output"],
        )
        self.llm_analyzer = LLMAnalyzer(self.llm_config)
        self.llm_analyzer.storage_config = self.config

        # Storage
        self.storage = PodcastStorage(storage_config=self.config)

    def run(self, youtube_url: str) -> dict:
        """Run full pipeline on a YouTube URL."""
        logger.info(f"Starting pipeline for: {youtube_url}")

        # Step 1: Download audio
        audio_result = self.downloader.download_audio(youtube_url)
        logger.info(f"Downloaded: {audio_result.audio_path}")

        # Step 2: Transcribe
        transcript_result = self.transcriber.transcribe(
            audio_path=audio_result.audio_path,
            title=audio_result.title,
        )
        logger.info(f"Transcribed: {len(transcript_result.text)} chars")

        # Step 3: Diarize
        diarization_result = self.diarizer.diarize(
            audio_path=audio_result.audio_path,
            transcript=transcript_result,
        )
        logger.info(f"Diarized: {len(diarization_result.speakers)} speakers")

        # Step 4: Combine transcript + diarization and save to file
        transcript = {
            "video_title": audio_result.title,
            "raw_text": transcript_result.text,
            "segments": transcript_result.segments,
            "speaker_segments": diarization_result.speaker_segments,
            "speakers": diarization_result.speakers,
        }
        transcript_path = self.transcriber.save_transcript(
            title=audio_result.title,
            raw_text=transcript_result.text,
            segments=transcript_result.segments,
            speaker_segments=diarization_result.speaker_segments,
            speakers=diarization_result.speakers,
        )
        logger.info(f"Transcript saved: {transcript_path}")

        # Step 5: LLM analysis
        summary_path = self.llm_analyzer.analyze(
            title=audio_result.title,
            transcript=transcript,
            analysis_type="summary",
        )
        logger.info(f"Summary saved: {summary_path}")

        insights_path = self.llm_analyzer.analyze(
            title=audio_result.title,
            transcript=transcript,
            analysis_type="insights",
        )
        logger.info(f"Insights saved: {insights_path}")

        # Step 6: Store in database
        podcast_id = self.storage.add_podcast(
            title=audio_result.title,
            url=youtube_url,
            video_id=audio_result.video_id,
            audio_path=audio_result.audio_path,
            transcript_path=transcript_path,
            summary_path=summary_path,
            insights_path=insights_path,
        )
        logger.info(f"Podcast stored with ID: {podcast_id}")

        return {
            "podcast_id": podcast_id,
            "audio_path": audio_result.audio_path,
            "transcript_path": transcript_path,
            "summary_path": summary_path,
            "insights_path": insights_path,
        }

    def analyze_existing(self, podcast_id: int) -> dict:
        """Analyze an existing podcast from the database."""
        podcast = self.storage.get_podcast(podcast_id)
        if not podcast:
            logger.error(f"Podcast ID {podcast_id} not found")
            return None

        transcript = self.storage.get_transcript(podcast_id)
        if not transcript:
            logger.error(f"Transcript for podcast ID {podcast_id} not found")
            return None

        summary_path = self.llm_analyzer.analyze(
            title=podcast["title"],
            transcript=transcript,
            analysis_type="summary",
        )
        insights_path = self.llm_analyzer.analyze(
            title=podcast["title"],
            transcript=transcript,
            analysis_type="insights",
        )

        self.storage.update_podcast(podcast_id, summary_path=summary_path, insights_path=insights_path)

        return {
            "podcast_id": podcast_id,
            "summary_path": summary_path,
            "insights_path": insights_path,
        }


@dataclass
class LLMAnalyzerConfig:
    """Configuration for LLM analyzer."""
    provider: str = "lmstudio"
    model: str = "qwen/qwen3.6-35b-a3b"
    lmstudio_url: str = "http://localhost:1234"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: int = 300
    streaming: bool = False
    output_format: str = "markdown"
    enable_structured_output: bool = True
