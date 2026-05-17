"""YouTube audio downloader module — yt-dlp wrapper for podcast audio extraction."""

import subprocess
import json
import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Structured metadata extracted from YouTube video."""
    video_id: str
    title: str
    description: str
    channel: str
    channel_id: str
    uploader: str
    upload_date: str
    tags: list[str]
    categories: list[str]
    duration: Optional[float]
    view_count: Optional[int]
    like_count: Optional[int]
    thumbnail_url: Optional[str]


@dataclass
class AudioDownloadResult:
    """Result of an audio download operation."""
    video_id: str
    title: str
    audio_path: str
    metadata: VideoMetadata
    duration: Optional[float]
    success: bool
    error: Optional[str]


class YouTubeAudioDownloader:
    """Wrapper around yt-dlp for extracting audio from YouTube videos with metadata."""

    def __init__(
        self,
        audio_format: str = "mp3",
        audio_quality: str = "best",
        output_dir: str = "data/audio",
    ):
        self.audio_format = audio_format
        self.audio_quality = audio_quality
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _parse_info_json(self, info_path: str) -> VideoMetadata:
        """Parse yt-dlp info.json file into structured metadata."""
        with open(info_path, "r") as f:
            data = json.load(f)

        return VideoMetadata(
            video_id=data.get("id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            channel=data.get("channel", ""),
            channel_id=data.get("channel_id", ""),
            uploader=data.get("uploader", ""),
            upload_date=data.get("upload_date", ""),
            tags=data.get("tags", []),
            categories=data.get("categories", []),
            duration=data.get("duration"),
            view_count=data.get("view_count"),
            like_count=data.get("like_count"),
            thumbnail_url=data.get("thumbnail"),
        )

    def _find_matching_files(self, video_id: str) -> tuple[str, str]:
        """Find the audio file and info.json matching a specific video_id."""
        audio_files = [
            f for f in os.listdir(self.output_dir) if f.endswith(f".{self.audio_format}")
        ]
        info_files = [
            f for f in os.listdir(self.output_dir) if f.endswith(".info.json")
        ]

        if not audio_files or not info_files:
            return "", ""

        # Match by video_id in info.json
        for info_file in info_files:
            info_path = os.path.join(self.output_dir, info_file)
            try:
                with open(info_path, "r") as f:
                    info_data = json.load(f)
                if info_data.get("id") == video_id:
                    # Find corresponding audio file (same base name)
                    base_name = info_file.replace(".info.json", "")
                    audio_file = f"{base_name}.{self.audio_format}"
                    if audio_file in audio_files:
                        return (
                            os.path.join(self.output_dir, audio_file),
                            info_path,
                        )
            except (json.JSONDecodeError, IOError):
                continue

        # Fallback: use the most recent files
        return (
            os.path.join(self.output_dir, audio_files[-1]),
            os.path.join(self.output_dir, info_files[-1]),
        )

    def download_audio(self, url: str) -> AudioDownloadResult:
        """Download audio from a YouTube video URL with full metadata."""
        output_template = os.path.join(
            self.output_dir, "%(title)s.%(ext)s"
        )

        cmd = [
            "yt-dlp",
            "--extract-audio",
            f"--audio-format={self.audio_format}",
            f"--audio-quality={self.audio_quality}",
            f"--output={output_template}",
            "--no-playlist",
            "--write-info-json",
            "--no-overwrites",
            url,
        ]

        logger.info(f"Downloading audio from: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            logger.error(f"Download failed: {error_msg}")
            return AudioDownloadResult(
                video_id="",
                title="",
                audio_path="",
                metadata=VideoMetadata(
                    video_id="",
                    title="",
                    description="",
                    channel="",
                    channel_id="",
                    uploader="",
                    upload_date="",
                    tags=[],
                    categories=[],
                    duration=None,
                    view_count=None,
                    like_count=None,
                    thumbnail_url=None,
                ),
                duration=None,
                success=False,
                error=error_msg,
            )

        # Parse info.json to get video_id, then find matching files
        info_files = [
            f for f in os.listdir(self.output_dir) if f.endswith(".info.json")
        ]
        if not info_files:
            return AudioDownloadResult(
                video_id="",
                title="",
                audio_path="",
                metadata=VideoMetadata(
                    video_id="",
                    title="",
                    description="",
                    channel="",
                    channel_id="",
                    uploader="",
                    upload_date="",
                    tags=[],
                    categories=[],
                    duration=None,
                    view_count=None,
                    like_count=None,
                    thumbnail_url=None,
                ),
                duration=None,
                success=False,
                error="No info.json found after download",
            )

        # Get video_id from most recent info.json
        info_path = os.path.join(self.output_dir, info_files[-1])
        metadata = self._parse_info_json(info_path)
        video_id = metadata.video_id

        audio_path, matched_info_path = self._find_matching_files(video_id)
        if not audio_path:
            return AudioDownloadResult(
                video_id=video_id,
                title=metadata.title,
                audio_path="",
                metadata=metadata,
                duration=metadata.duration,
                success=False,
                error="No audio file found after download",
            )

        logger.info(f"Downloaded: {metadata.title} ({video_id})")
        return AudioDownloadResult(
            video_id=video_id,
            title=metadata.title,
            audio_path=audio_path,
            metadata=metadata,
            duration=metadata.duration,
            success=True,
            error=None,
        )

    def download_from_channel(
        self, channel_id: str, limit: int = 5
    ) -> list[AudioDownloadResult]:
        """Download recent audio from a YouTube channel."""
        url = f"https://www.youtube.com/channel/{channel_id}/videos"

        cmd = [
            "yt-dlp",
            "--extract-audio",
            f"--audio-format={self.audio_format}",
            f"--audio-quality={self.audio_quality}",
            f"--output={os.path.join(self.output_dir, '%(title)s.%(ext)s')}",
            f"--playlist-limit={limit}",
            "--write-info-json",
            "--no-overwrites",
            url,
        ]

        logger.info(f"Downloading up to {limit} videos from channel: {channel_id}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            logger.error(f"Channel download failed: {error_msg}")
            return []

        # Parse all info.json files to build results
        info_files = [
            f for f in os.listdir(self.output_dir) if f.endswith(".info.json")
        ]
        audio_files = [
            f for f in os.listdir(self.output_dir) if f.endswith(f".{self.audio_format}")
        ]

        results = []
        for info_file in info_files:
            info_path = os.path.join(self.output_dir, info_file)
            try:
                with open(info_path, "r") as f:
                    info_data = json.load(f)
                video_id = info_data.get("id", "")
                metadata = self._parse_info_json(info_path)

                # Find matching audio file
                base_name = info_file.replace(".info.json", "")
                audio_file = f"{base_name}.{self.audio_format}"
                audio_path = os.path.join(self.output_dir, audio_file) if audio_file in audio_files else ""

                if audio_path:
                    results.append(AudioDownloadResult(
                        video_id=video_id,
                        title=metadata.title,
                        audio_path=audio_path,
                        metadata=metadata,
                        duration=metadata.duration,
                        success=True,
                        error=None,
                    ))
            except (json.JSONDecodeError, IOError):
                continue

        logger.info(f"Downloaded {len(results)} videos from channel: {channel_id}")
        return results
