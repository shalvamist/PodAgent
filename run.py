#!/usr/bin/env python3
"""PodAgent — Main entry point for podcast processing pipeline."""

import argparse
import yaml
import logging
import os
import torch

logger = logging.getLogger(__name__)
from src.downloader import YouTubeAudioDownloader
from src.transcriber import WhisperTranscriber
from src.diarizer import SpeakerDiarizer
from src.transcript_builder import TranscriptBuilder
from src.channel_monitor import ChannelMonitor
from src.storage import PodcastStorage


def load_config(config_path="config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    log_level = config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("file", None)
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers
    )


def process_single_video(url: str, config: dict, storage: PodcastStorage):
    """Process a single YouTube video through the full pipeline with GPU detection and metadata context."""
    # GPU detection at start
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_mem:.1f}GB VRAM")
        if gpu_mem < 6:
            logger.warning(f"GPU has <6GB VRAM ({gpu_mem:.1f}GB) — switching to medium model instead of turbo")
            model_name = config["settings"]["transcription"]["model"]
            if model_name == "turbo":
                model_name = "medium"
                config["settings"]["transcription"]["model"] = model_name
    else:
        logger.info("No GPU available — running on CPU (fp32 mode, ~2-3x slower)")

    downloader = YouTubeAudioDownloader(
        audio_format=config["settings"]["audio_format"],
        output_dir=config["settings"]["storage"]["audio_dir"]
    )
    transcriber = WhisperTranscriber(
        model=config["settings"]["transcription"]["model"],
        language=config["settings"]["transcription"]["language"],
        carry_initial_prompt=True,  # Carry context prompt across all windows
    )
    diarizer = SpeakerDiarizer(
        hf_token=config["settings"]["diarization"]["hf_token"]
    )
    builder = TranscriptBuilder(
        transcript_dir=config["settings"]["storage"]["transcript_dir"]
    )

    # Step 1: Download audio + metadata
    download_result = downloader.download_audio(url)
    if not download_result.success:
        logger.error(f"Download failed: {download_result.error}")
        return

    metadata = download_result.metadata
    logger.info(f"Video: {metadata.title}")
    logger.info(f"Channel: {metadata.channel}")
    logger.info(f"Uploader: {metadata.uploader}")
    logger.info(f"Duration: {metadata.duration}s")
    logger.info(f"Tags: {metadata.tags}")

    # Step 2: Transcribe with context prompt from metadata
    transcription_result = transcriber.transcribe(
        download_result.audio_path,
        video_info={
            "title": metadata.title,
            "description": metadata.description,
            "channel": metadata.channel,
            "channel_id": metadata.channel_id,
            "uploader": metadata.uploader,
            "tags": metadata.tags,
        }
    )
    if not transcription_result.success:
        logger.error(f"Transcription failed: {transcription_result.error}")
        return

    # Step 3: Diarize
    diarization_result = diarizer.diarize(download_result.audio_path)
    if not diarization_result.success:
        logger.error(f"Diarization failed: {diarization_result.error}")
        return

    # Step 4: Build structured transcript with metadata context
    transcript = builder.build(
        transcription_result, diarization_result,
        video_title=metadata.title,
        metadata=metadata,
        podcaster_speaker=None  # Auto-detect: first speaker = uploader
    )

    # Step 5: Save to storage
    storage.save_podcast({
        "video_id": metadata.video_id,
        "title": metadata.title,
        "channel_id": metadata.channel_id,
        "channel_name": metadata.channel,
        "audio_path": download_result.audio_path,
        "transcript_path": transcript.output_path,
        "language": transcription_result.language,
        "duration": transcription_result.duration,
        "num_speakers": diarization_result.num_speakers
    })
    storage.save_segments(1, transcript.segments)
    storage.save_speakers(1, transcript.speakers)

    logger.info(f"Podcast processed: {metadata.title}")
    logger.info(f"Speakers identified: {len(transcript.speakers)}")
    for sp in transcript.speakers:
        logger.info(f"  {sp.speaker_id} -> {sp.label}")


def main():
    parser = argparse.ArgumentParser(description="PodAgent — YouTube podcast pipeline")
    parser.add_argument("--url", help="Process a single YouTube video URL")
    parser.add_argument("--monitor", action="store_true", help="Monitor all channels for new uploads")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    storage = PodcastStorage(
        db_path=os.path.join(config["settings"]["storage"]["audio_dir"], "..", "podagent.db")
    )

    if args.url:
        process_single_video(args.url, config, storage)
    elif args.monitor:
        monitor = ChannelMonitor(
            channels_file="data/channels.yaml",
            storage_dir=config["settings"]["storage"]["transcript_dir"]
        )
        new_videos = monitor.monitor_all_channels()
        logging.info(f"Found {len(new_videos)} new videos")
        for video in new_videos:
            url = f"https://www.youtube.com/watch?v={video['video_id']}"
            process_single_video(url, config, storage)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
