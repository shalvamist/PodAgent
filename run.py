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
from src.audio_chunker import split_audio_into_chunks, cleanup_chunks, get_audio_duration
from src import utils


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
        dir_path = os.path.dirname(log_file)
        if dir_path:  # Only create dir if path has a directory component
            os.makedirs(dir_path, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def validate_config(config: dict) -> list[str]:
    """Validate critical config fields. Returns list of warning messages."""
    warnings = []

    # LLM provider / URL consistency check
    llm = config.get("settings", {}).get("llm", {})
    provider = llm.get("provider", "")
    if provider == "ollama":
        base_url = llm.get("base_url", "")
        if base_url and str(11434) not in base_url:
            warnings.append(
                f"Provider is 'ollama' but base_url '{base_url}' does not "
                f"contain expected port 11434. Did you mean "
                f"http://localhost:11434?"
            )
    elif provider == "lmstudio":
        lmstudio_url = llm.get("lmstudio_url", "")
        if not lmstudio_url:
            warnings.append(
                "Provider is 'lmstudio' but no lmstudio_url configured. "
                "LLM analysis will fail."
            )

    diarization = config.get("settings", {}).get("diarization", {})
    hf_token = diarization.get("hf_token", "")
    if not hf_token or "YOUR_HF_TOKEN" in hf_token:
        warnings.append(
            "HuggingFace token is missing or placeholder — diarization will fail. "
            "Set settings.diarization.hf_token in config.yaml."
        )

    channels = [c for c in config.get("channels", []) if isinstance(c, dict)]
    if not channels:
        warnings.append(
            "No valid channels configured in config.yaml — channel monitoring will be empty."
        )

    transcription = config.get("settings", {}).get("transcription", {})
    model = transcription.get("model", "")
    valid_models = {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "turbo"}
    if model and model not in valid_models:
        warnings.append(f"Unknown Whisper model '{model}' — may fail at runtime.")

    return warnings


def _resolve_yt_dlp_path(config: dict) -> str:
    """Resolve yt-dlp path from config — relative paths resolved against project root."""
    raw = config.get("settings", {}).get("yt_dlp_path", ".venv/bin/yt-dlp")
    if not os.path.isabs(raw):
        # Resolve relative to the directory where run.py lives (project root)
        project_root = os.path.dirname(os.path.abspath(__file__))
        raw = os.path.join(project_root, raw)
    return raw


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


def _process_chunk(
    chunk, transcriber, diarizer, metadata_dict, video_title, video_id,
):
    """Process a single audio chunk through transcription + diarization.

    Returns (transcription_result, diarization_result) or (None, None) on failure.
    """
    # Transcribe this chunk with context prompt from metadata
    transcription_result = transcriber.transcribe(
        chunk.audio_path,
        video_info=metadata_dict,
    )
    if not transcription_result.success:
        logger.error(f"Transcription failed for chunk {chunk.chunk_index + 1}: {transcription_result.error}")
        return None, None

    # Diarize this chunk
    diarization_result = diarizer.diarize(chunk.audio_path)
    if not diarization_result.success:
        logger.error(f"Diarization failed for chunk {chunk.chunk_index + 1}: {diarization_result.error}")
        return None, None

    return transcription_result, diarization_result


def _merge_chunk_results(
    chunks, chunk_results, metadata, video_title, video_id, video_folder,
):
    """Merge per-chunk transcription and diarization results into a single transcript.

    Applies time offsets so all segment timestamps are relative to the original video start.
    Deduplicates speakers across chunks by matching labels (first speaker = podcaster).
    Uses the provided video_folder (generated at download time) — does NOT create new folders.
    """
    from src.transcript_builder import SpeakerProfile, StructuredTranscript
    from src.folder_manager import sanitize_filename
    from src.utils import assign_speaker_labels

    # Collect all segments with time offsets and track unique speakers
    merged_segments = []
    speaker_label_map = {}  # maps pyannote_id -> canonical label

    for chunk, (trans_result, diag_result) in zip(chunks, chunk_results):
        if trans_result is None or diag_result is None:
            continue

        offset = chunk.start_time

        # Assign speaker labels using shared helper (extends global map incrementally)
        speaker_label_map = assign_speaker_labels(
            diag_result.speaker_segments, metadata, speaker_label_map,
        )

        # Merge transcription segments with diarization labels, applying time offset
        for trans_seg in trans_result.segments:
            pyannote_id = None
            for diag_seg in diag_result.speaker_segments:
                if diag_seg["start"] <= trans_seg["start"] < diag_seg["end"]:
                    pyannote_id = diag_seg["speaker"]
                    break

            # Fallback: if no exact match, find closest diarization segment
            if pyannote_id is None and diag_result.speaker_segments:
                best_dist = float('inf')
                for diag_seg in diag_result.speaker_segments:
                    dist = min(abs(trans_seg["start"] - diag_seg["start"]), abs(trans_seg["end"] - diag_seg["end"]))
                    if dist < best_dist:
                        best_dist = dist
                        pyannote_id = diag_seg["speaker"]

            label = speaker_label_map.get(pyannote_id, "Unknown")
            merged_segments.append({
                "start": round(trans_seg["start"] + offset, 1),
                "end": round(trans_seg["end"] + offset, 1),
                "speaker_label": label,
                "text": trans_seg["text"],
            })

    # Post-processing: reassign any remaining Unknown segments via majority vote of neighbors
    for i in range(len(merged_segments)):
        if merged_segments[i]["speaker_label"] == "Unknown":
            # Look at up to 3 preceding and 3 succeeding segments with known speakers
            window = []
            for j in range(max(0, i - 5), min(len(merged_segments), i + 6)):
                if j != i and merged_segments[j]["speaker_label"] != "Unknown":
                    # Weight by proximity (closer segments get higher weight)
                    dist = abs(j - i)
                    window.append((merged_segments[j]["speaker_label"], 1.0 / max(dist, 1)))

            if window:
                # Majority vote weighted by proximity
                votes = {}
                for label, weight in window:
                    votes[label] = votes.get(label, 0) + weight
                if votes:
                    best_label = max(votes.keys(), key=lambda k: votes[k])
                    merged_segments[i]["speaker_label"] = best_label

    # Build speaker profiles from merged data (deduplicated)
    seen_speaker_ids = set()
    speakers = []
    for seg in merged_segments:
        label = seg["speaker_label"]
        if label not in seen_speaker_ids:
            seen_speaker_ids.add(label)
            first_appearance = seg["start"]
            speakers.append(SpeakerProfile(
                speaker_id=label,  # Use the canonical label as ID for merged results
                label=label,
                first_appearance=first_appearance,
            ))

    # Sort speakers by first appearance
    speakers.sort(key=lambda s: s.first_appearance)

    # Calculate total duration from last segment
    total_duration = max((seg["end"] for seg in merged_segments), default=0)

    # Build raw text from all segments
    raw_text = " ".join(seg["text"] for seg in merged_segments)

    # Save to Markdown file — use the SAME folder generated at download time (video_folder)
    transcript_folder = os.path.join(video_folder, "transcript")
    os.makedirs(transcript_folder, exist_ok=True)

    sanitized_title = sanitize_filename(video_title)
    output_path = os.path.join(
        transcript_folder,
        f"{sanitized_title}_transcript.md",
    )

    md_lines = []
    md_lines.append(f"# {video_title}")
    md_lines.append("")
    md_lines.append(f"**Duration:** {total_duration:.1f}s")
    md_lines.append(f"**Language:** {chunk_results[0][0].language if chunk_results and chunk_results[0][0] else 'en'}")
    md_lines.append("")
    md_lines.append("## Speakers")
    md_lines.append("")
    for s in speakers:
        first_appearance_min = s.first_appearance / 60
        md_lines.append(f"- **{s.label}** (first at {first_appearance_min:.1f}m)")
    md_lines.append("")
    md_lines.append("## Transcript")
    md_lines.append("")
    for seg in merged_segments:
        md_lines.append(f"**{seg['speaker_label']}**: {seg['text']}")
    md_lines.append("")

    md_content = "\n".join(md_lines)
    with open(output_path, "w") as f:
        f.write(md_content)

    logger.info(f"Transcript saved to: {output_path}")
    logger.info(f"Merged speakers: {len(speakers)}, segments: {len(merged_segments)}")

    # Collect per-chunk segment data for analysis (preserves time range info)
    chunk_segment_data = []
    seg_idx = 0
    for chunk, (trans_result, diag_result) in zip(chunks, chunk_results):
        if trans_result is None or diag_result is None:
            continue
        offset = chunk.start_time
        # Use the global label map — already built from all chunks above
        local_label_map = speaker_label_map.copy()
        chunk_segs = []
        for trans_seg in trans_result.segments:
            pyannote_id = None
            for diag_seg in diag_result.speaker_segments:
                if diag_seg["start"] <= trans_seg["start"] < diag_seg["end"]:
                    pyannote_id = diag_seg["speaker"]
                    break
            if pyannote_id is None and diag_result.speaker_segments:
                best_dist = float('inf')
                for diag_seg in diag_result.speaker_segments:
                    dist = min(abs(trans_seg["start"] - diag_seg["start"]), abs(trans_seg["end"] - diag_seg["end"]))
                    if dist < best_dist:
                        best_dist = dist
                        pyannote_id = diag_seg["speaker"]
            label = local_label_map.get(pyannote_id, "Unknown")
            chunk_segs.append({
                "start": round(trans_seg["start"] + offset, 1),
                "end": round(trans_seg["end"] + offset, 1),
                "speaker_label": label,
                "text": trans_seg["text"],
            })
        chunk_segment_data.append({
            "chunk_index": chunk.chunk_index,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "segments": chunk_segs,
        })

    return StructuredTranscript(
        video_title=video_title,
        audio_path=chunk_results[0][0].audio_path if chunk_results and chunk_results[0][0] else "",
        language=chunk_results[0][0].language if chunk_results and chunk_results[0][0] else "en",
        duration=total_duration,
        speakers=speakers,
        segments=merged_segments,
        raw_text=raw_text,
        output_path=output_path,
    ), chunk_segment_data


def process_single_video(url: str, config: dict, storage: PodcastStorage, analyze: bool = False, tts_source: object = None):
    """Process a single YouTube video through the full pipeline.

    For videos longer than 1 hour (3600s), splits audio into chunks and processes each one.
    All output goes into ONE per-video folder generated at download time.
    LLM analysis is always run on the FULL merged transcript, never per-chunk.
    """
    gpu_mode = detect_gpu(config)

    # Determine whisper model based on GPU
    transcriber_model = config["settings"]["transcription"]["model"]
    if gpu_mode == "medium_cpu":
        transcriber_model = "medium"
        config["settings"]["transcription"]["model"] = transcriber_model

    downloader = YouTubeAudioDownloader(
        audio_format=config["settings"]["audio_format"],
        base_data_dir=config["settings"]["storage"]["audio_dir"].replace("/audio", ""),
        yt_dlp_path=_resolve_yt_dlp_path(config),
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

    # Step 1: Download audio + metadata
    download_result = downloader.download_audio(url)
    if not download_result.success:
        logger.error(f"Download failed: {download_result.error}")
        return None

    metadata = download_result.metadata
    logger.info("CHECKPOINT: DOWNLOAD_COMPLETE")
    logger.info(f"Video: {metadata.title}")
    logger.info(f"Channel: {metadata.channel}")
    logger.info(f"Uploader: {metadata.uploader}")
    logger.info(f"Duration: {metadata.duration}s")
    logger.info(f"Tags: {metadata.tags}")

    # Step 1b: Check if audio needs chunking (> 1 hour)
    actual_duration = get_audio_duration(download_result.audio_path) or metadata.duration or 0
    max_chunk_seconds = config.get("settings", {}).get("transcription", {}).get("max_chunk_seconds", 3600)

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

        logger.info(f"CHECKPOINT: CHUNKING_COMPLETE — {len(chunks)} chunks created")

    # Step 2/3: Transcribe and diarize (single file or per-chunk)
    metadata_dict = {
        "title": metadata.title,
        "description": metadata.description,
        "channel": metadata.channel,
        "channel_id": metadata.channel_id,
        "uploader": metadata.uploader,
        "tags": metadata.tags,
    }

    if chunks:
        # Process each chunk independently
        logger.info(f"Processing {len(chunks)} audio chunks...")
        chunk_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"--- Chunk {i+1}/{len(chunks)} [{chunk.start_time:.0f}s - {chunk.end_time:.0f}s] ---")
            trans_result, diag_result = _process_chunk(
                chunk, transcriber, diarizer, metadata_dict,
                metadata.title, metadata.video_id,
            )
            if trans_result is None or diag_result is None:
                logger.error(f"Chunk {i+1} failed — aborting pipeline")
                cleanup_chunks(chunks)
                return None
            chunk_results.append((trans_result, diag_result))
            logger.info(f"CHECKPOINT: CHUNK_{i+1}_COMPLETE — transcribed and diarized")

        # Merge all chunks into a single transcript (reuses video_folder from download)
        logger.info("Merging chunk results...")
        transcript, chunk_segment_data = _merge_chunk_results(
            chunks, chunk_results, metadata,
            metadata.title, metadata.video_id, download_result.video_folder,
        )

        # Clean up temporary chunk files
        cleanup_chunks(chunks)
        logger.info("CHECKPOINT: MERGE_COMPLETE")

        # Use merged data for storage
        total_duration = transcript.duration
        num_speakers = len(transcript.speakers)
        avg_confidence = None
        if chunk_results:
            confs = [r[0].confidence for r in chunk_results if r[0].confidence is not None]
            avg_confidence = sum(confs) / len(confs) if confs else None

    else:
        # Original single-file path (unchanged)
        logger.info("Audio within duration limit — processing as single file")
        chunk_segment_data = []  # no chunks for single-file processing

        # Step 2: Transcribe with context prompt from metadata
        transcription_result = transcriber.transcribe(
            download_result.audio_path,
            video_info=metadata_dict,
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

        total_duration = transcription_result.duration
        num_speakers = diarization_result.num_speakers

    # Step 5: Save to storage with quality metrics (both chunked and single-file)
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
        "transcription_confidence": avg_confidence if chunks else (transcription_result.confidence if hasattr(transcription_result, "confidence") else None),
        "diarization_quality": 1.0 if chunks else (diarization_result.quality if hasattr(diarization_result, "quality") else None),
    })
    storage.save_segments(podcast_id, transcript.segments)
    storage.save_speakers(podcast_id, transcript.speakers)

    logger.info("CHECKPOINT: STORAGE_COMPLETE")
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
        )
        analyzer = LLMAnalyzer(analyzer_config)

        is_available, error_msg = analyzer.check_availability()
        if is_available:
            logger.info(f"LLM available: {analyzer_config.provider} ({analyzer_config.model})")
            available_models = analyzer.list_available_models()
            logger.info(f"Available models: {available_models}")

            modes = ["summary", "insights", "notes", "blog"]
            base_data_dir = config["settings"]["storage"]["audio_dir"].replace("/audio", "")

            # Always analyze the FULL merged transcript — never per-chunk.
            # Chunks are only for Whisper transcription; LLM sees the whole thing.
            if chunk_segment_data:
                logger.info(
                    f"Video was chunked into {len(chunk_segment_data)} segments for Whisper, "
                    f"but analysis runs on the full merged transcript."
                )

            transcript_data = {
                "video_title": transcript.video_title,
                "speakers": [
                    {"speaker_id": s.speaker_id, "label": s.label, "first_appearance": s.first_appearance}
                    for s in transcript.speakers
                ],
                "segments": transcript.segments,
                "raw_text": transcript.raw_text,
                "video_id": metadata.video_id,
            }

            for mode in modes:
                logger.info(f"Running LLM analysis: mode={mode}")
                result = analyzer.analyze(transcript_data, mode=mode, base_data_dir=base_data_dir, video_folder=download_result.video_folder)
                if result.summary_text.startswith("ERROR"):
                    logger.warning(f"LLM analysis failed for mode={mode}: {result.summary_text}")
                else:
                    logger.info(f"LLM analysis complete: mode={mode}, time={result.processing_time_seconds:.2f}s")
                    analysis_dict = result.__dict__.copy()
                    storage.save_llm_analysis(podcast_id, analysis_dict)
                    analyses.append(analysis_dict)
                    storage.update_search_index(podcast_id)

            analyzer.close()
            logger.info("CHECKPOINT: ANALYSIS_COMPLETE")
        else:
            logger.error(f"LLM analysis skipped — {error_msg}")

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

        tts_source_text = None
        tts_source_label = None

        if isinstance(tts_source, str):
            tts_file_path = tts_source
            if os.path.isfile(tts_file_path):
                with open(tts_file_path, "r") as f:
                    tts_source_text = f.read()
                tts_source_label = f"custom file: {tts_file_path}"
                logger.info(f"Reading TTS source from: {tts_file_path}")
            else:
                logger.warning(f"TTS file not found: {tts_file_path}")
        elif tts_source is True:
            if analyses and len(analyses) > 0:
                summary_text = analyses[0].get("summary_text", "")
                if summary_text and not summary_text.startswith("ERROR"):
                    tts_source_text = summary_text
                    tts_source_label = "LLM summary"
                else:
                    logger.info("No summary text available for TTS")
            else:
                logger.info("No LLM analyses available for TTS")

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

    # Validate config and warn about issues (don't block — let it fail gracefully)
    config_warnings = validate_config(config)
    if config_warnings:
        logger.warning("Configuration issues detected:")
        for w in config_warnings:
            logger.warning(f"  - {w}")

    storage_dir = config["settings"]["storage"]["audio_dir"].replace("/audio", "")
    db_path = os.path.join(storage_dir, "podagent.db")
    storage = PodcastStorage(db_path=db_path)

    if args.url:
        if not utils.validate_url(args.url):
            logger.error(f"Invalid YouTube URL: {args.url}")
            sys.exit(1)
        process_single_video(args.url, config, storage, analyze=args.analyze, tts_source=args.tts)

    elif args.monitor:
        monitor = ChannelMonitor(
            channels_file="data/channels.yaml",
            storage_dir=config["settings"]["storage"]["audio_dir"],
            yt_dlp_path=config.get("settings", {}).get("yt_dlp_path"),
        )
        fetch_results = monitor.monitor_all_channels()
        # Flatten results, skipping failed channels
        new_videos = []
        for fr in fetch_results:
            if fr.error:
                logger.warning(f"Skipping channel {fr.channel_id} (error: {fr.error})")
            else:
                new_videos.extend(fr.new_videos)
        logger.info(f"Found {len(new_videos)} new videos across all channels")
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
