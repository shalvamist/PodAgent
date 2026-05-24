"""Channel monitor module — monitors YouTube channels for new podcast uploads."""

import yaml
import json
import os
import logging
import subprocess
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class ChannelMonitor:
    """Monitor YouTube channels for new podcast uploads."""

    def __init__(
        self,
        channels_file: str = "data/channels.yaml",
        storage_dir: str = "data/transcripts",
        poll_interval: str = "6h",
    ):
        self.channels_file = channels_file
        self.storage_dir = storage_dir
        self.poll_interval = poll_interval

        # Load channel list
        self.channels = self._load_channels()

        # Track last processed timestamps
        self._processed_log = os.path.join(
            self.storage_dir, "processed_log.json"
        )
        self._processed_videos = self._load_processed_log()

    def _load_channels(self) -> list[dict]:
        """Load channel list from YAML file."""
        if not os.path.exists(self.channels_file):
            logger.warning(f"Channels file not found: {self.channels_file}")
            return []

        with open(self.channels_file, "r") as f:
            data = yaml.safe_load(f)
        return data.get("channels", [])

    def _load_processed_log(self) -> dict:
        """Load log of already-processed videos."""
        if os.path.exists(self._processed_log):
            with open(self._processed_log, "r") as f:
                return json.load(f)
        return {}

    def _save_processed_log(self):
        """Save processed video log."""
        with open(self._processed_log, "w") as f:
            json.dump(self._processed_videos, f, indent=2)

    def find_new_videos(self, channel_id: str, limit: int = 20) -> list[dict]:
        """Find new videos from a channel that haven't been processed."""
        url = f"https://www.youtube.com/channel/{channel_id}/videos"

        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--print", "%(id)s|%(title)s|%(upload_date)s|%(uploader)s|%(channel)s",
            "--playlist-reverse",
            f"--playlist-limit={limit}",
            url,
        ]

        logger.info(f"Fetching new videos from channel: {channel_id}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            logger.error(f"Channel fetch failed: {error_msg}")
            return []

        new_videos = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 5:
                video_id = parts[0]
                title = parts[1]
                upload_date = parts[2]
                uploader = parts[3]
                channel = parts[4]

                if video_id not in self._processed_videos:
                    new_videos.append({
                        "video_id": video_id,
                        "title": title,
                        "upload_date": upload_date,
                        "uploader": uploader,
                        "channel": channel,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                    })

        logger.info(f"Found {len(new_videos)} new videos from channel: {channel_id}")
        return new_videos

    def get_channel_info(self, channel_id: str) -> dict:
        """Extract full channel metadata."""
        url = f"https://www.youtube.com/channel/{channel_id}"

        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--print", "%(channel)s|%(channel_id)s|%(uploader)s|%(description)s",
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"Channel info fetch failed: {result.stderr.strip()}")
            return {}

        parts = result.stdout.strip().split("|")
        if len(parts) >= 4:
            return {
                "channel": parts[0],
                "channel_id": parts[1],
                "uploader": parts[2],
                "description": parts[3],
            }

        return {}

    def monitor_all_channels(self) -> list[dict]:
        """Find new videos across all configured channels."""
        new_videos = []
        for channel in self.channels:
            channel_id = channel["id"]
            channel_name = channel.get("name", channel_id)
            logger.info(f"Monitoring channel: {channel_name}")
            new = self.find_new_videos(channel_id)
            new_videos.extend(new)

        logger.info(f"Total new videos found: {len(new_videos)}")
        return new_videos

    def mark_video_processed(self, video_id: str):
        """Mark a video as processed in the log."""
        self._processed_videos[video_id] = {
            "processed_at": datetime.now().isoformat(),
        }
        self._save_processed_log()

    def process_new_videos(self, new_videos: list[dict], config: dict = None):
        """Queue and process new videos through the full pipeline."""
        from src.downloader import YouTubeAudioDownloader
        from src.transcriber import WhisperTranscriber
        from src.diarizer import SpeakerDiarizer
        from src.transcript_builder import TranscriptBuilder
        from src.storage import PodcastStorage

        if not new_videos:
            logger.info("No new videos to process")
            return

        logger.info(f"Processing {len(new_videos)} new videos")

        # Initialize pipeline components
        if config:
            downloader = YouTubeAudioDownloader(
                audio_format=config["settings"]["audio_format"],
                base_data_dir=config["settings"]["storage"]["audio_dir"],
            )
            transcriber = WhisperTranscriber(
                model=config["settings"]["transcription"]["model"],
                language=config["settings"]["transcription"]["language"],
                carry_initial_prompt=config["settings"]["transcription"].get(
                    "carry_initial_prompt", True
                ),
            )
            diarizer = SpeakerDiarizer(
                hf_token=config["settings"]["diarization"]["hf_token"],
            )
            builder = TranscriptBuilder(
                base_data_dir=config["settings"]["storage"]["audio_dir"],
            )
            storage = PodcastStorage(
                db_path=os.path.join(
                    config["settings"]["storage"]["audio_dir"],
                    "podagent.db"
                ),
            )
        else:
            downloader = YouTubeAudioDownloader()
            transcriber = WhisperTranscriber()
            diarizer = SpeakerDiarizer()
            builder = TranscriptBuilder()
            storage = PodcastStorage()

        for video in new_videos:
            logger.info(f"Processing video: {video['title']}")
            try:
                # Step 1: Download
                download_result = downloader.download_audio(video["url"])
                if not download_result.success:
                    logger.error(f"Download failed: {download_result.error}")
                    continue

                # Step 2: Transcribe
                transcription_result = transcriber.transcribe(
                    download_result.audio_path,
                    video_info={
                        "title": download_result.metadata.title,
                        "description": download_result.metadata.description,
                        "channel": download_result.metadata.channel,
                        "channel_id": download_result.metadata.channel_id,
                        "uploader": download_result.metadata.uploader,
                        "tags": download_result.metadata.tags,
                    },
                )
                if not transcription_result.success:
                    logger.error(f"Transcription failed: {transcription_result.error}")
                    continue

                # Step 3: Diarize
                diarization_result = diarizer.diarize(download_result.audio_path)
                if not diarization_result.success:
                    logger.error(f"Diarization failed: {diarization_result.error}")
                    continue

                # Step 4: Build transcript
                transcript = builder.build(
                    transcription_result,
                    diarization_result,
                    video_title=download_result.metadata.title,
                    video_id=download_result.metadata.video_id,
                    metadata=download_result.metadata,
                    podcaster_speaker=None,
                )

                # Step 5: Save to storage
                podcast_id = storage.save_podcast({
                    "video_id": download_result.metadata.video_id,
                    "title": download_result.metadata.title,
                    "channel_id": download_result.metadata.channel_id,
                    "channel_name": download_result.metadata.channel,
                    "audio_path": download_result.audio_path,
                    "transcript_path": transcript.output_path,
                    "language": transcription_result.language,
                    "duration": transcription_result.duration,
                    "num_speakers": diarization_result.num_speakers,
                })
                storage.save_segments(podcast_id, transcript.segments)
                storage.save_speakers(podcast_id, transcript.speakers)

                # Mark as processed
                self.mark_video_processed(download_result.metadata.video_id)

                logger.info(f"Video processed: {download_result.metadata.title}")

            except Exception as e:
                logger.error(f"Processing failed for {video['title']}: {e}")
                continue

        logger.info(f"Completed processing {len(new_videos)} videos")
