"""PodAgent pipeline — single source of truth for the full processing pipeline.

This module provides importable functions that encapsulate the complete
YouTube podcast processing workflow: download, transcribe, diarize, build
transcript, save to storage, and optionally analyze with LLM or generate TTS.

Usage from run.py CLI:
    transcript = process_video(url, config_path="config.yaml", analyze=True)

Usage from channel monitor:
    results = monitor_channels(config_path="config.yaml")
"""

import os
import sys
import logging
from typing import Optional, Union, List, Any, Literal

logger = logging.getLogger(__name__)

AnalysisMode = Literal["summary", "insights", "notes"]
ALL_ANALYSIS_MODES: list[AnalysisMode] = ["summary", "insights", "notes"]


def _resolve_project_root() -> str:
    """Resolve the project root directory (where run.py and config.yaml live)."""
    return os.path.dirname(os.path.abspath(__file__))  # src/ → parent is project root


def process_video(
    url: str,
    config_path: str = "config.yaml",
    analysis_modes: Optional[list[AnalysisMode]] = None,
    tts_source=None,
    skip_download: bool = False,
) -> Optional[object]:
    """Process a single YouTube video through the full pipeline.

    For videos longer than max_chunk_seconds (default 1800s), splits audio
    into chunks and processes each one independently before merging results.
    LLM analysis runs on the FULL merged transcript, never per-chunk.

    Args:
        url: YouTube video URL.
        config_path: Path to config.yaml (relative or absolute).
        analyze: If True, run LLM analysis after transcription.
        tts_source: If True, generate TTS from LLM summary; if a string,
            treat as file path for custom text source.
        skip_download: If True, reuse existing audio from a previous run.

    Returns:
        StructuredTranscript on success, None on failure.
    """
    project_root = _resolve_project_root()

    # Resolve config path relative to project root if needed
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    from src.downloader import YouTubeAudioDownloader
    from src.transcriber import WhisperTranscriber
    from src.diarizer import SpeakerDiarizer
    from src.transcript_builder import TranscriptBuilder
    from src.storage import PodcastStorage
    from src.llm_analyzer import LLMAnalyzer, LLMAnalyzerConfig
    from src.tts_generator import TTSGenerator, TTSConfig
    from src.audio_chunker import split_audio_into_chunks, cleanup_chunks, get_audio_duration

    # Load config
    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return None

    # Resolve yt-dlp path relative to project root
    raw_ytdlp = config.get("settings", {}).get("yt_dlp_path", ".venv/bin/yt-dlp")
    if not os.path.isabs(raw_ytdlp):
        raw_ytdlp = os.path.join(project_root, raw_ytdlp)

    # GPU detection and Whisper model selection
    import torch
    gpu_mode = "cuda_turbo"
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_mem:.1f}GB VRAM")
        min_vram = config["settings"]["gpu"]["min_vram_gb"]
        if gpu_mem < min_vram:
            gpu_mode = "medium_cpu"
    else:
        logger.info("No GPU available — running on CPU (fp32 mode, ~2-3x slower)")

    transcriber_model = config["settings"]["transcription"]["model"]
    if gpu_mode == "medium_cpu":
        transcriber_model = "medium"

    # Initialize components
    storage_dir = config["settings"]["storage"]["audio_dir"].replace("/audio", "")
    db_path = os.path.join(storage_dir, "podagent.db")
    storage = PodcastStorage(db_path=db_path)

    downloader = YouTubeAudioDownloader(
        audio_format=config["settings"]["audio_format"],
        base_data_dir=storage_dir,
        yt_dlp_path=raw_ytdlp,
        js_runtime_path=config["settings"].get("js_runtime_path"),
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

    # Step 1: Download audio + metadata (or reuse existing)
    if skip_download:
        from run import _extract_video_id_from_url
        video_id = _extract_video_id_from_url(url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {url}")
            return None
        logger.info(f"Skipping download — looking for existing audio for video_id={video_id}")
        download_result = downloader.find_existing_audio(video_id)
    else:
        download_result = downloader.download_audio(url)

    if not download_result.success:
        logger.error(f"Download failed: {download_result.error}")
        return None

    metadata = download_result.metadata
    logger.info("CHECKPOINT: DOWNLOAD_COMPLETE")
    logger.info(f"Video: {metadata.title}, Duration: {metadata.duration}s")

    # Determine if we need to chunk the audio
    actual_duration = get_audio_duration(download_result.audio_path) or metadata.duration or 0
    max_chunk_seconds = config.get("settings", {}).get(
        "transcription", {}
    ).get("max_chunk_seconds", 1800)

    chunks = []
    if actual_duration > max_chunk_seconds:
        logger.info(
            f"Audio is {actual_duration:.0f}s — exceeds {max_chunk_seconds}s limit. "
            f"Splitting into {int(actual_duration / max_chunk_seconds)} chunk(s)"
        )
        chunks = split_audio_into_chunks(
            download_result.audio_path,
            chunk_duration_seconds=max_chunk_seconds,
        )
        if not chunks:
            logger.error("Failed to split audio — aborting")
            return None

    # Step 2/3: Transcribe and diarize
    metadata_dict = {
        "title": metadata.title,
        "description": metadata.description,
        "channel": metadata.channel,
        "channel_id": metadata.channel_id,
        "uploader": metadata.uploader,
        "tags": metadata.tags,
    }

    if chunks:
        # Chunked processing path
        logger.info(f"Processing {len(chunks)} audio chunks...")

        chunk_results = []
        for i, chunk in enumerate(chunks):
            logger.info(
                f"--- Chunk {i+1}/{len(chunks)} [{chunk.start_time:.0f}s - "
                f"{chunk.end_time:.0f}s] ---"
            )
            trans_result = transcriber.transcribe(chunk.audio_path, video_info=metadata_dict)
            if not trans_result.success:
                logger.error(
                    f"Transcription failed for chunk {i+1}: {trans_result.error}"
                )
                cleanup_chunks(chunks)
                return None

            diag_result = diarizer.diarize(chunk.audio_path)
            if not diag_result.success:
                logger.error(
                    f"Diarization failed for chunk {i+1}: {diag_result.error}"
                )
                cleanup_chunks(chunks)
                return None

            chunk_results.append((trans_result, diag_result))
            logger.info(f"CHECKPOINT: CHUNK_{i+1}_COMPLETE")

        # Import merge function from run.py (shared logic)
        from run import _merge_chunk_results
        transcript, chunk_segment_data = _merge_chunk_results(
            chunks, chunk_results, metadata,
            metadata.title, metadata.video_id, download_result.video_folder,
        )
        cleanup_chunks(chunks)

        total_duration = transcript.duration
        num_speakers = len(transcript.speakers)
    else:
        # Single-file processing path
        logger.info("Audio within duration limit — processing as single file")
        chunk_segment_data = []

        transcription_result = transcriber.transcribe(
            download_result.audio_path, video_info=metadata_dict
        )
        if not transcription_result.success:
            logger.error(f"Transcription failed: {transcription_result.error}")
            return None

        diarization_result = diarizer.diarize(download_result.audio_path)
        if not diarization_result.success:
            logger.error(f"Diarization failed: {diarization_result.error}")
            return None

        transcript = builder.build(
            transcription_result,
            diarization_result,
            video_title=metadata.title,
            video_id=metadata.video_id,
            metadata=metadata,
        )
        total_duration = transcription_result.duration
        num_speakers = diarization_result.num_speakers

    # Step 5: Save to storage
    podcast_id = storage.save_podcast({
        "video_id": metadata.video_id,
        "title": metadata.title,
        "channel_id": metadata.channel_id,
        "channel_name": metadata.channel,
        "audio_path": download_result.audio_path,
        "transcript_path": transcript.output_path,
        "language": transcript.language,
        "duration": total_duration,
        "num_speakers": num_speakers,
    })
    storage.save_segments(podcast_id, transcript.segments)
    storage.save_speakers(podcast_id, transcript.speakers)

    logger.info("CHECKPOINT: STORAGE_COMPLETE")
    logger.info(f"Podcast processed: {metadata.title}")
    logger.info(f"Speakers identified: {num_speakers}")

    # Step 6: LLM analysis (optional) — default all modes when requested
    if analysis_modes and config.get("settings", {}).get("llm"):
        llm_config = config["settings"]["llm"]
        analyzer_config = LLMAnalyzerConfig(
            provider=llm_config.get("provider", "ollama"),
            model=llm_config.get("model", "llama3"),
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            lmstudio_url=llm_config.get("lmstudio_url", "http://localhost:1234"),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 4096),
            timeout_seconds=llm_config.get("timeout_seconds", 120),
        )
        analyzer = LLMAnalyzer(analyzer_config)

        if analyzer.check_availability()[0]:
            for mode in analysis_modes:
                logger.info(f"Running LLM analysis: mode={mode}")
                transcript_data = {
                    "video_title": transcript.video_title,
                    "speakers": [
                        {"speaker_id": s.speaker_id, "label": s.label,
                         "first_appearance": s.first_appearance}
                        for s in transcript.speakers
                    ],
                    "segments": transcript.segments,
                    "raw_text": transcript.raw_text,
                    "video_id": metadata.video_id,
                }
                result = analyzer.analyze(
                    transcript_data, mode=mode,
                    base_data_dir=storage_dir,
                    video_folder=download_result.video_folder,
                )
                if not result.summary_text.startswith("ERROR"):
                    analysis_dict = result.__dict__.copy()
                    storage.save_llm_analysis(podcast_id, analysis_dict)
                    storage.update_search_index(podcast_id)

            analyzer.close()
            logger.info("CHECKPOINT: ANALYSIS_COMPLETE")

    # Step 7: TTS generation (optional)
    if tts_source and config.get("settings", {}).get("tts"):
        tts_config_data = config["settings"]["tts"]
        tts_generator = TTSGenerator(TTSConfig(
            provider=tts_config_data.get("provider", "edge-tts"),
            voice=tts_config_data.get("voice", "en-US-AvaMultilingualNeural"),
            rate=tts_config_data.get("rate", "+0%"),
            pitch=tts_config_data.get("pitch", "+0Hz"),
            output_format=tts_config_data.get("output_format", "mp3"),
        ))

        if tts_source is True:
            analyses = storage.get_llm_analyses(podcast_id)
            if analyses and len(analyses) > 0:
                summary_text = analyses[0].get("summary_text", "")
                if summary_text and not summary_text.startswith("ERROR"):
                    tts_dir = os.path.join(
                        os.path.dirname(os.path.dirname(transcript.output_path)), "tts"
                    )
                    os.makedirs(tts_dir, exist_ok=True)
                    tts_output_path = os.path.join(
                        tts_dir, f"{metadata.video_id}_tts.mp3"
                    )
                    tts_result = tts_generator.generate(summary_text, tts_output_path)
                    if tts_result["success"]:
                        logger.info(f"TTS audio generated: {tts_result['output_path']}")

    return transcript


def monitor_channels(
    config_path: str = "config.yaml",
    analysis_modes: Optional[list[AnalysisMode]] = None,
    tts_source=None,
) -> list[Any]:
    """Monitor all configured YouTube channels for new uploads and process them.

    Each channel is processed independently — a failure on one channel does not
    prevent others from being checked (error isolation).

    Args:
        config_path: Path to config.yaml (relative or absolute).
        analyze: If True, run LLM analysis after transcription.
        tts_source: TTS source option (see process_video for details).

    Returns:
        List of ChannelFetchResult with per-channel status and new videos found.
    """
    project_root = _resolve_project_root()

    # Resolve config path
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return []

    # Resolve channels file path relative to project root
    channels_file = config.get("settings", {}).get(
        "channels_file", os.path.join(project_root, "data/channels.yaml")
    )
    if not os.path.isabs(channels_file):
        channels_file = os.path.join(project_root, channels_file)

    from src.channel_monitor import ChannelMonitor

    monitor = ChannelMonitor(
        channels_file=channels_file,
        storage_dir=config["settings"]["storage"]["audio_dir"],
        yt_dlp_path=config.get("settings", {}).get("yt_dlp_path"),
    )

    fetch_results = monitor.monitor_all_channels()

    # Process new videos through the pipeline
    for fr in fetch_results:
        if fr.error:
            logger.warning(
                f"Skipping channel {fr.channel_id} (error: {fr.error})"
            )
            continue

        for video in fr.new_videos:
            url = f"https://www.youtube.com/watch?v={video['video_id']}"
            process_video(url, config_path=config_path, analysis_modes=analysis_modes, tts_source=tts_source)

    return list(fetch_results)  # type: ignore[return-value]
