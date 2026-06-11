"""YouTube audio downloader module — yt-dlp wrapper for podcast audio extraction."""

import subprocess
import json
import os
import logging
import shutil
from dataclasses import dataclass
from typing import Optional

from src.utils import retry


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
        base_data_dir: str = "data",
        yt_dlp_path: str = ".venv/bin/yt-dlp",
        js_runtime_path: Optional[str] = None,
    ):
        self.audio_format = audio_format
        self.audio_quality = audio_quality
        self.base_data_dir = base_data_dir
        self.yt_dlp_path = yt_dlp_path
        self.js_runtime_path = js_runtime_path  # Full path to deno (or None for PATH lookup)
        os.makedirs(base_data_dir, exist_ok=True)

    def _get_video_folder(self, title: str) -> str:
        """Get or create the per-video data folder using structured naming."""
        from src.folder_manager import get_video_folder
        return get_video_folder(self.base_data_dir, title)

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

    @retry(max_attempts=3, delay=2)
    def download_audio(self, url: str) -> AudioDownloadResult:
        """Download audio from a YouTube video URL with full metadata."""
        logger.info(f"Downloading audio from: {url}")

        # Run yt-dlp first to get video_id from info.json
        output_template = os.path.join(self.base_data_dir, "%(title)s.%(ext)s")
        # Determine JS runtime argument for n-challenge solving
        if self.js_runtime_path:
            js_runtime_arg = ["deno"]  # yt-dlp only accepts names, not full paths
            # Ensure deno is on PATH via environment variable
            env = os.environ.copy()
            deno_dir = os.path.dirname(os.path.expanduser(self.js_runtime_path))
            existing_path = env.get("PATH", "")
            if deno_dir not in existing_path:
                env["PATH"] = f"{deno_dir}:{existing_path}"
        else:
            js_runtime_arg = ["deno"]
            env = None

        cmd = [
            self.yt_dlp_path,
            "--cookies-from-browser", "chrome",
            "--js-runtimes", *js_runtime_arg,
            "--extract-audio",
            f"--audio-format={self.audio_format}",
            f"--audio-quality={self.audio_quality}",
            f"--output={output_template}",
            "--no-playlist",
            "--write-info-json",
            "--no-overwrites",
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

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

        # Find the most recent info.json to get video_id
        info_files = [
            f for f in os.listdir(self.base_data_dir) if f.endswith(".info.json")
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
        info_path = os.path.join(self.base_data_dir, info_files[-1])
        metadata = self._parse_info_json(info_path)
        video_id = metadata.video_id

        # Create per-video folder and move files into it
        video_folder = self._get_video_folder(metadata.title)
        audio_folder = os.path.join(video_folder, "audio")
        os.makedirs(audio_folder, exist_ok=True)

        # Find matching audio file
        base_name = info_files[-1].replace(".info.json", "")
        audio_file = f"{base_name}.{self.audio_format}"

        # Move info.json into audio folder
        shutil.move(info_path, os.path.join(audio_folder, info_files[-1]))

        # Move audio file into audio folder if it exists
        audio_path = ""
        if audio_file in os.listdir(self.base_data_dir):
            audio_path = os.path.join(audio_folder, audio_file)
            shutil.move(
                os.path.join(self.base_data_dir, audio_file),
                audio_path
            )
        else:
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
        logger.info(f"Files stored in: {video_folder}")
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

        output_template = os.path.join(self.base_data_dir, "%(title)s.%(ext)s")
        cmd = [
            self.yt_dlp_path,
            "--extract-audio",
            f"--audio-format={self.audio_format}",
            f"--audio-quality={self.audio_quality}",
            f"--output={output_template}",
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
            f for f in os.listdir(self.base_data_dir) if f.endswith(".info.json")
        ]
        audio_files = [
            f for f in os.listdir(self.base_data_dir) if f.endswith(f".{self.audio_format}")
        ]

        results = []
        for info_file in info_files:
            info_path = os.path.join(self.base_data_dir, info_file)
            try:
                with open(info_path, "r") as f:
                    info_data = json.load(f)
                video_id = info_data.get("id", "")
                metadata = self._parse_info_json(info_path)

                # Find matching audio file
                base_name = info_file.replace(".info.json", "")
                audio_file = f"{base_name}.{self.audio_format}"
                audio_path = os.path.join(self.base_data_dir, audio_file) if audio_file in audio_files else ""

                if audio_path:
                    # Create per-video folder and move files into it
                    video_folder = self._get_video_folder(metadata.title)
                    audio_folder = os.path.join(video_folder, "audio")
                    os.makedirs(audio_folder, exist_ok=True)

                    # Move info.json into audio folder
                    shutil.move(info_path, os.path.join(audio_folder, info_file))

                    # Move audio file into audio folder
                    new_audio_path = os.path.join(audio_folder, audio_file)
                    shutil.move(audio_path, new_audio_path)

                    results.append(AudioDownloadResult(
                        video_id=video_id,
                        title=metadata.title,
                        audio_path=new_audio_path,
                        metadata=metadata,
                        duration=metadata.duration,
                        success=True,
                        error=None,
                    ))
            except (json.JSONDecodeError, IOError):
                continue

        logger.info(f"Downloaded {len(results)} videos from channel: {channel_id}")
        return results
