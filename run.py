#!/usr/bin/env python3
"""PodAgent — Main entry point for podcast processing pipeline."""

import argparse
import yaml
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

from src.downloader import YouTubeAudioDownloader
from src.transcriber import WhisperTranscriber
from src.diarizer import SpeakerDiarizer
from src.transcript_builder import TranscriptBuilder
from src.channel_monitor import ChannelMonitor
from src.storage import PodcastStorage
from src.llm_analyzer import LLMAnalyzer, LLMAnalyzerConfig
from src.tts_generator import TTSGenerator, TTSConfig


def load_config(config_path="config.yaml") -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Configure logging from config."""
    log_level = config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("file", None)
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    global logger
    logger = logging.getLogger(__name__)


def detect_gpu(config: dict) -> str:
    """Detect GPU availability and return inference mode."""
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_mem:.1f}GB VRAM")
        if gpu_mem < config["settings"]["gpu"]["min_vram_gb"]:
            logger.warning(
                f"GPU has <{config['settings']['gpu']['min_vram_gb']}GB VRAM "
                f"({gpu_mem:.1f}GB) — switching to medium model instead of turbo"
            )
            return "medium_cpu"
        return "cuda_turbo"
    else:
        logger.info("No GPU available — running on CPU (fp32 mode, ~2-3x slower)")
        return "cpu"


def process_single_video(url: str, config: dict, storage: PodcastStorage, analyze: bool = False, tts_source: object = None):
    """Process a single YouTube video through the full pipeline."""
    gpu_mode = detect_gpu(config)

    # Determine whisper model based on GPU
    transcriber_model = config["settings"]["transcription"]["model"]
    if gpu_mode == "medium_cpu":
        transcriber_model = "medium"
        config["settings"]["transcription"]["model"] = transcriber_model

    downloader = YouTubeAudioDownloader(
        audio_format=config["settings"]["audio_format"],
        base_data_dir=config["settings"]["storage"]["audio_dir"].replace("/audio", ""),
        yt_dlp_path=os.path.join(os.path.dirname(__file__), ".venv/bin/yt-dlp"),
    )
    transcriber = WhisperTranscriber(
        model=transcriber_model,
        language=config["settings"]["transcription"]["language"],
        carry_initial_prompt=True,
    )
    diarizer = SpeakerDiarizer(
        hf_token=config["settings"]["diarization"]["hf_token"],
    )
    builder = TranscriptBuilder(
        base_data_dir=config["settings"]["storage"]["audio_dir"],
    )

    # Step 1: Download audio + metadata
    download_result = downloader.download_audio(url)
    if not download_result.success:
        logger.error(f"Download failed: {download_result.error}")
        return None

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
        },
    )
    if not transcription_result.success:
        logger.error(f"Transcription failed: {transcription_result.error}")
        return None

    # Step 3: Diarize
    diarization_result = diarizer.diarize(download_result.audio_path)
    if not diarization_result.success:
        logger.error(f"Diarization failed: {diarization_result.error}")
        return None

    # Step 4: Build structured transcript with metadata context
    transcript = builder.build(
        transcription_result,
        diarization_result,
        video_title=metadata.title,
        video_id=metadata.video_id,
        metadata=metadata,
        podcaster_speaker=None,
    )

    # Step 5: Save to storage with quality metrics
    podcast_id = storage.save_podcast({
        "video_id": metadata.video_id,
        "title": metadata.title,
        "channel_id": metadata.channel_id,
        "channel_name": metadata.channel,
        "audio_path": download_result.audio_path,
        "transcript_path": transcript.output_path,
        "language": transcription_result.language,
        "duration": transcription_result.duration,
        "num_speakers": diarization_result.num_speakers,
        "transcription_confidence": transcription_result.confidence if hasattr(transcription_result, "confidence") else None,
        "diarization_quality": diarization_result.quality if hasattr(diarization_result, "quality") else None,
    })
    storage.save_segments(podcast_id, transcript.segments)
    storage.save_speakers(podcast_id, transcript.speakers)

    logger.info(f"Podcast processed: {metadata.title}")
    logger.info(f"Speakers identified: {len(transcript.speakers)}")
    for sp in transcript.speakers:
        logger.info(f"  {sp.speaker_id} -> {sp.label}")

    # Step 6: LLM analysis (optional)
    analyses = []
    if analyze:
        llm_config = config.get("settings", {}).get("llm", LLMAnalyzerConfig().__dict__)
        analyzer_config = LLMAnalyzerConfig(
            provider=llm_config.get("provider", "ollama"),
            model=llm_config.get("model", "llama3"),
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            lmstudio_url=llm_config.get("lmstudio_url", "http://localhost:1234"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 4096),
            timeout_seconds=llm_config.get("timeout_seconds", 120),
            streaming=llm_config.get("streaming", True),
            enable_structured_output=llm_config.get("enable_structured_output", True),
        )
        analyzer = LLMAnalyzer(analyzer_config)

        # Check availability
        analyses = []
        if analyzer.check_availability():
            logger.info(f"LLM available: {analyzer_config.provider} ({analyzer_config.model})")
            available_models = analyzer.list_available_models()
            logger.info(f"Available models: {available_models}")

            # Build transcript dict from in-memory data (no disk read needed)
            transcript_data = {
                "video_title": transcript.video_title,
                "speakers": [
                    {
                        "speaker_id": s.speaker_id,
                        "label": s.label,
                        "first_appearance": s.first_appearance,
                    }
                    for s in transcript.speakers
                ],
                "segments": transcript.segments,
                "raw_text": transcript.raw_text,
                "video_id": metadata.video_id,
            }

            # Run analysis in all modes
            modes = ["summary", "insights", "notes", "blog"]
            base_data_dir = config["settings"]["storage"]["audio_dir"].replace("/audio", "")
            analyses = []
            for mode in modes:
                logger.info(f"Running LLM analysis: mode={mode}")
                result = analyzer.analyze(
                    transcript_data,
                    mode=mode,
                    base_data_dir=base_data_dir,
                )
                if result.summary_text.startswith("ERROR"):
                    logger.warning(f"LLM analysis failed for mode={mode}: {result.summary_text}")
                else:
                    logger.info(f"LLM analysis complete: mode={mode}, time={result.processing_time_seconds:.2f}s")
                    # Save to storage with structured fields
                    storage.save_llm_analysis(podcast_id, result.__dict__)
                    analyses.append(result.__dict__)
                    # Update FTS5 search index
                    storage.update_search_index(podcast_id)

            analyzer.close()
        else:
            logger.warning(f"LLM service unavailable at {analyzer_config.base_url}")
            logger.info("Skipping LLM analysis — start Ollama/LM Studio to enable it")

    # Step 7: TTS generation (optional)
    if tts_source and config.get("settings", {}).get("tts"):
        tts_config_data = config["settings"]["tts"]
        tts_config = TTSConfig(
            provider=tts_config_data.get("provider", "edge-tts"),
            voice=tts_config_data.get("voice", "en-US-AvaMultilingualNeural"),
            rate=tts_config_data.get("rate", "+0%"),
            pitch=tts_config_data.get("pitch", "+0Hz"),
            output_format=tts_config_data.get("output_format", "mp3"),
            elevenlabs_api_key=tts_config_data.get("elevenlabs_api_key", ""),
            elevenlabs_model=tts_config_data.get("elevenlabs_model", "eleven_multilingual_v2"),
        )
        tts_generator = TTSGenerator(tts_config)

        # Determine source text for TTS
        tts_source_text = None
        tts_source_label = None

        if isinstance(tts_source, str):
            # Custom file path provided
            tts_file_path = tts_source
            if os.path.isfile(tts_file_path):
                with open(tts_file_path, "r") as f:
                    tts_source_text = f.read()
                tts_source_label = f"custom file: {tts_file_path}"
                logger.info(f"Reading TTS source from: {tts_file_path}")
            else:
                logger.warning(f"TTS file not found: {tts_file_path}")
        elif tts_source is True:
            # Use LLM summary from analysis
            if analyses and len(analyses) > 0:
                summary_text = analyses[0].get("summary_text", "")
                if summary_text and not summary_text.startswith("ERROR"):
                    tts_source_text = summary_text
                    tts_source_label = "LLM summary"
                else:
                    logger.info("No summary text available for TTS")
            else:
                logger.info("No LLM analyses available for TTS")

        # Generate TTS if we have source text
        if tts_source_text:
            tts_dir = os.path.join(os.path.dirname(os.path.dirname(transcript.output_path)), "tts")
            os.makedirs(tts_dir, exist_ok=True)
            tts_output_path = os.path.join(tts_dir, f"{metadata.video_id}_tts.mp3")
            logger.info(f"Generating TTS from {tts_source_label}: {tts_output_path}")
            tts_result = tts_generator.generate(tts_source_text, tts_output_path)
            if tts_result["success"]:
                logger.info(f"TTS audio generated: {tts_result['output_path']} ({tts_result['file_size']/1024:.1f}KB)")
                tts_result["provider"] = tts_config.provider
                tts_result["voice"] = tts_config.voice
                tts_result["source"] = tts_source_label
                storage.save_tts_audio(podcast_id, tts_result)
            else:
                logger.warning(f"TTS generation failed: {tts_result['error']}")

    return transcript


def main():
    parser = argparse.ArgumentParser(
        description="PodAgent — YouTube podcast downloader, transcriber, and diarizer",
    )
    parser.add_argument("--url", help="Process a single YouTube video URL")
    parser.add_argument("--monitor", action="store_true", help="Monitor all channels for new uploads")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run LLM analysis on transcripts (requires Ollama/LM Studio)",
    )
    parser.add_argument(
        "--tts",
        nargs="?",
        const=True,
        default=None,
        help="Generate TTS audio. Use --tts (no arg) for LLM summary, or --tts <file_path> for custom file",
    )
    parser.add_argument(
        "--list-analyses",
        action="store_true",
        help="List all LLM analyses stored in database",
    )
    parser.add_argument(
        "--list-tts",
        action="store_true",
        help="List all TTS audio records stored in database",
    )
    parser.add_argument(
        "--list-podcasts",
        action="store_true",
        help="List all podcasts stored in database",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    storage_dir = config["settings"]["storage"]["audio_dir"].replace("/audio", "")
    db_path = os.path.join(storage_dir, "podagent.db")
    storage = PodcastStorage(db_path=db_path)

    if args.url:
        process_single_video(args.url, config, storage, analyze=args.analyze, tts_source=args.tts)

    elif args.monitor:
        monitor = ChannelMonitor(
            channels_file="data/channels.yaml",
            storage_dir=config["settings"]["storage"]["audio_dir"],
        )
        new_videos = monitor.monitor_all_channels()
        logger.info(f"Found {len(new_videos)} new videos")
        for video in new_videos:
            url = f"https://www.youtube.com/watch?v={video['video_id']}"
            process_single_video(url, config, storage, analyze=args.analyze, tts_source=args.tts)

    elif args.list_analyses:
        analyses = storage.get_llm_analyses()
        if not analyses:
            logger.info("No LLM analyses stored")
        else:
            logger.info(f"LLM analyses ({len(analyses)}):")
            for a in analyses:
                logger.info(
                    f"  podcast_id={a['podcast_id']} mode={a['analysis_mode']} "
                    f"model={a['llm_model']} provider={a['provider']} "
                    f"time={a['processing_time']:.2f}s"
                )

    elif args.list_tts:
        tts_records = storage.get_tts_audio()
        if not tts_records:
            logger.info("No TTS audio records stored")
        else:
            logger.info(f"TTS audio records ({len(tts_records)}):")
            for t in tts_records:
                logger.info(
                    f"  podcast_id={t['podcast_id']} provider={t['tts_provider']} "
                    f"voice={t['voice']} path={t['audio_path']} "
                    f"size={t['file_size']/1024:.1f}KB"
                )

    elif args.list_podcasts:
        podcasts = storage.get_all_podcasts()
        if not podcasts:
            logger.info("No podcasts stored")
        else:
            logger.info(f"Podcasts ({len(podcasts)}):")
            for p in podcasts:
                logger.info(
                    f"  id={p['id']} title={p['title']} channel={p['channel_name']} "
                    f"duration={p['duration']:.0f}s speakers={p['num_speakers']}"
                )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
