"""Audio chunking module — split long audio into manageable pieces for transcription."""

import subprocess
import os
import logging
from dataclasses import dataclass
from typing import Optional, List


logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents a single chunk of audio with its boundaries and path."""
    start_time: float  # Start time in seconds (relative to original)
    end_time: float  # End time in seconds (relative to original)
    chunk_index: int  # Zero-based index
    audio_path: str  # Path to the temporary chunk file


def get_audio_duration(audio_path: str) -> Optional[float]:
    """Get duration of an audio file using ffprobe.

    Returns duration in seconds, or None if unavailable.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            logger.debug(f"Audio duration: {duration:.1f}s ({os.path.basename(audio_path)})")
            return duration
    except FileNotFoundError:
        logger.warning("ffprobe not found — cannot determine audio duration")
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
    return None


def split_audio_into_chunks(
    audio_path: str, chunk_duration_seconds: float = 3600
) -> List[AudioChunk]:
    """Split an audio file into equal-duration chunks using ffmpeg.

    Args:
        audio_path: Path to the original audio file (e.g., MP3).
        chunk_duration_seconds: Maximum duration of each chunk in seconds.

    Returns:
        List of AudioChunk objects, or empty list on failure.
    """
    if not os.path.isfile(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return []

    # Get total duration
    total_duration = get_audio_duration(audio_path)
    if total_duration is None:
        logger.error("Could not determine audio duration — cannot chunk")
        return []

    logger.info(
        f"Splitting '{os.path.basename(audio_path)}' ({total_duration:.0f}s) "
        f"into chunks of {chunk_duration_seconds}s"
    )

    # Calculate number of chunks needed
    num_chunks = max(1, int(total_duration / chunk_duration_seconds)) + 1
    if total_duration <= chunk_duration_seconds:
        logger.info("Audio is within duration limit — no splitting needed")
        return []

    output_dir = os.path.dirname(audio_path) or "."
    chunks = []

    # Use ffmpeg to split audio into segments
    # Pattern: {output_dir}/chunk_{{index}}_{start:.0f}s.mp3
    for i in range(num_chunks):
        start_time = i * chunk_duration_seconds
        if start_time >= total_duration:
            break

        end_time = min(start_time + chunk_duration_seconds, total_duration)
        filename = f"chunk_{i}_{int(start_time)}s.mp3"
        output_path = os.path.join(output_dir, filename)

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",  # overwrite existing files
                    "-ss", str(start_time),  # start time
                    "-i", audio_path,  # input file
                    "-t", str(end_time - start_time),  # duration of this chunk
                    "-vn",  # no video (audio only)
                    "-b:a", "192k",  # bit rate for output
                    "-map", "0:a:0",  # map first audio stream
                    output_path,
                ],
                capture_output=True, text=True, timeout=600
            )

            if result.returncode != 0:
                logger.error(
                    f"ffmpeg failed for chunk {i+1}: {result.stderr[:500]}"
                )
                # Clean up any partial files before returning
                cleanup_chunks(chunks)
                return []

            chunks.append(AudioChunk(
                start_time=start_time,
                end_time=end_time,
                chunk_index=i,
                audio_path=output_path,
            ))
            logger.info(f"  Created chunk {i+1}: [{start_time:.0f}s - {end_time:.0f}s] -> {os.path.basename(output_path)}")

        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg timed out splitting chunk {i+1}")
            cleanup_chunks(chunks)
            return []
        except Exception as e:
            logger.error(f"Error splitting audio at chunk {i+1}: {e}")
            cleanup_chunks(chunks)
            return []

    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def cleanup_chunks(chunks: List[AudioChunk]) -> None:
    """Remove temporary chunk files.

    Safely deletes each chunk file, logging any failures.
    """
    if not chunks:
        return

    for i, chunk in enumerate(chunks):
        try:
            if os.path.exists(chunk.audio_path):
                os.remove(chunk.audio_path)
                logger.debug(f"Removed chunk {i+1}: {os.path.basename(chunk.audio_path)}")
        except OSError as e:
            logger.error(f"Failed to remove chunk file {chunk.audio_path}: {e}")

    logger.info("Cleanup complete")


# ---- Standalone usage example ----
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python audio_chunker.py <audio_file.mp3> [chunk_duration_seconds]")
        sys.exit(1)

    audio_file = sys.argv[1]
    chunk_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 150

    chunks = split_audio_into_chunks(audio_file, chunk_duration)
    if chunks:
        print(f"\nCreated {len(chunks)} chunks:")
        for c in chunks:
            print(
                f"  [{c.chunk_index}] {c.start_time:.0f}s - {c.end_time:.0f}s -> "
                f"{os.path.basename(c.audio_path)}"
            )
    else:
        print("No chunks created.")
