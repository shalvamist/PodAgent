#!/usr/bin/env python3
"""Resume PodAgent pipeline from existing audio chunks — skips download step.

Usage:
    python3 run_resume.py --folder <output_folder_path> [--config config.yaml]
"""

import argparse, os, sys, logging, json, yaml, torch

# Resolve project root relative to this script's location so paths work
# regardless of where you invoke the command from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ResumePipeline")

from src.transcriber import WhisperTranscriber
from src.diarizer import SpeakerDiarizer
from src.audio_chunker import AudioChunk, cleanup_chunks
from src.storage import PodcastStorage
from src.folder_manager import generate_output_folder_name
from src.utils import assign_speaker_labels
from src.transcript_builder import SpeakerProfile, StructuredTranscript

def load_config(config_path):
    """Load config from the given path (absolute or relative to project root)."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_chunk(chunk, transcriber, diarizer, metadata_dict):
    trans_result = transcriber.transcribe(
        chunk.audio_path,
        video_info=metadata_dict,
    )
    if not trans_result.success:
        logger.error(f"Transcription failed for chunk {chunk.chunk_index}: {trans_result.error}")
        return None, None

    diag_result = diarizer.diarize(chunk.audio_path)
    if not diag_result.success:
        logger.error(f"Diarization failed for chunk {chunk.chunk_index}: {diag_result.error}")
        return None, None

    return trans_result, diag_result

def main():
    parser = argparse.ArgumentParser(description="Resume PodAgent from existing audio")
    parser.add_argument("--folder", required=True, help="Output folder path containing audio/")
    parser.add_argument("--config", default="config.yaml", help="Config file path (default: config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)

    # GPU detection
    gpu_mode = "cuda_turbo"
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_mem:.1f}GB VRAM")
        if gpu_mem < config["settings"]["gpu"]["min_vram_gb"]:
            gpu_mode = "medium_cpu"
    else:
        gpu_mode = "cpu"

    transcriber_model = config["settings"]["transcription"]["model"]
    if gpu_mode == "medium_cpu":
        transcriber_model = "medium"

    transcriber = WhisperTranscriber(
        model=transcriber_model,
        language=config["settings"]["transcription"]["language"],
        carry_initial_prompt=True,
    )
    diarizer = SpeakerDiarizer(
        hf_token=config["settings"]["diarization"]["hf_token"],
    )

    # Load metadata from info.json
    audio_dir = os.path.join(args.folder, "audio")
    if not os.path.isdir(audio_dir):
        logger.error(f"No audio/ directory found in {args.folder}")
        sys.exit(1)

    info_files = [f for f in os.listdir(audio_dir) if f.endswith(".info.json")]
    with open(os.path.join(audio_dir, info_files[0])) as f:
        info = json.load(f)

    metadata_dict = {
        "title": info.get("title", ""),
        "description": info.get("description", ""),
        "channel": info.get("channel", ""),
        "channel_id": info.get("channel_id", ""),
        "uploader": info.get("uploader", ""),
        "tags": info.get("tags", []),
    }

    video_title = info.get("title", "Unknown")
    video_id = info.get("id", "")

    # Build chunk list from existing files
    chunk_files = sorted(
        [f for f in os.listdir(audio_dir) if f.startswith("chunk_") and f.endswith(".mp3")]
    )

    chunks = []
    max_chunk_seconds = config.get("settings", {}).get("transcription", {}).get("max_chunk_seconds", 150)
    for cf in chunk_files:
        parts = cf.replace(".mp3", "").split("_")
        chunk_index = int(parts[1])
        start_time = float(parts[2].replace("s", ""))
        end_time = start_time + max_chunk_seconds
        chunks.append(AudioChunk(
            audio_path=os.path.join(audio_dir, cf),
            chunk_index=chunk_index,
            start_time=start_time,
            end_time=end_time,
        ))

    logger.info(f"Found {len(chunks)} chunks for '{video_title}'")

    # Process each chunk
    chunk_results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"--- Chunk {i+1}/{len(chunks)} [{chunk.start_time:.0f}s - {chunk.end_time:.0f}s] ---")
        trans_result, diag_result = process_chunk(chunk, transcriber, diarizer, metadata_dict)

        if trans_result is None or diag_result is None:
            logger.error(f"Chunk {i+1} failed — aborting pipeline")
            cleanup_chunks(chunks)
            sys.exit(1)

        chunk_results.append((trans_result, diag_result))
        logger.info(f"CHECKPOINT: CHUNK_{i+1}_COMPLETE")

    # Merge results
    merged_segments = []
    speaker_label_map = {}

    for chunk, (trans_result, diag_result) in zip(chunks, chunk_results):
        offset = chunk.start_time
        speaker_label_map = assign_speaker_labels(
            diag_result.speaker_segments, metadata_dict, speaker_label_map,
        )

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

            label = speaker_label_map.get(pyannote_id, "Unknown")
            merged_segments.append({
                "start": round(trans_seg["start"] + offset, 1),
                "end": round(trans_seg["end"] + offset, 1),
                "speaker_label": label,
                "text": trans_seg["text"],
            })

    # Post-process Unknown speakers via majority vote of neighbors
    for i in range(len(merged_segments)):
        if merged_segments[i]["speaker_label"] == "Unknown":
            window = []
            for j in range(max(0, i - 5), min(len(merged_segments), i + 6)):
                if j != i and merged_segments[j]["speaker_label"] != "Unknown":
                    dist = abs(j - i)
                    window.append((merged_segments[j]["speaker_label"], 1.0 / max(dist, 1)))

            if window:
                votes = {}
                for label, weight in window:
                    votes[label] = votes.get(label, 0) + weight
                best_label = max(votes.keys(), key=lambda k: votes[k])
                merged_segments[i]["speaker_label"] = best_label

    # Build speaker profiles
    seen_speaker_ids = set()
    speakers = []
    for seg in merged_segments:
        label = seg["speaker_label"]
        if label not in seen_speaker_ids:
            seen_speaker_ids.add(label)
            first_appearance = seg["start"]
            speakers.append(SpeakerProfile(
                speaker_id=label,
                label=label,
                first_appearance=first_appearance,
            ))

    speakers.sort(key=lambda s: s.first_appearance)
    total_duration = max((seg["end"] for seg in merged_segments), default=0)
    raw_text = " ".join(seg["text"] for seg in merged_segments)

    # Write transcript markdown
    video_folder = args.folder
    transcript_folder = os.path.join(video_folder, "transcript")
    os.makedirs(transcript_folder, exist_ok=True)

    safe_name = generate_output_folder_name(video_title)
    output_path = os.path.join(transcript_folder, f"{safe_name}_transcript.md")

    md_lines = [f"# {video_title}\n", f"**Duration:** {total_duration:.1f}s\n"]
    if chunk_results:
        md_lines.append(f"**Language:** {chunk_results[0][0].language or 'en'}\n")
    else:
        md_lines.append("**Language:** en\n")

    md_lines.append("\n## Speakers\n")
    for s in speakers:
        first_min = s.first_appearance / 60
        md_lines.append(f"- **{s.label}** (first at {first_min:.1f}m)\n")

    md_lines.append("\n## Transcript\n")
    for seg in merged_segments:
        md_lines.append(f"**{seg['speaker_label']}**: {seg['text']}\n")

    with open(output_path, "w") as f:
        f.write("".join(md_lines))

    logger.info(f"Transcript saved to: {output_path}")
    logger.info(f"Merged speakers: {len(speakers)}, segments: {len(merged_segments)}")

    # Save to database
    storage = PodcastStorage(db_path=os.path.join(PROJECT_ROOT, "data/podagent.db"))
    audio_file = [f for f in os.listdir(audio_dir) if f.endswith(".mp3") and not f.startswith("chunk_")]
    audio_path = os.path.join(audio_dir, audio_file[0]) if audio_file else ""

    podcast_id = storage.save_podcast({
        "video_id": video_id,
        "title": video_title,
        "channel_id": info.get("channel_id", ""),
        "channel_name": info.get("channel", ""),
        "audio_path": audio_path,
        "transcript_path": output_path,
        "language": chunk_results[0][0].language if chunk_results else "en",
        "duration": total_duration,
        "num_speakers": len(speakers),
    })

    storage.save_segments(podcast_id, merged_segments)
    storage.save_speakers(podcast_id, speakers)
    logger.info(f"CHECKPOINT: STORAGE_COMPLETE (id={podcast_id})")

    # Clean up chunks
    cleanup_chunks(chunks)
    logger.info("Chunk files cleaned up")

if __name__ == "__main__":
    main()
