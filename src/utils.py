"""Shared utilities for PodAgent."""

import os
import logging
import re
import time
import functools
from datetime import datetime


logger = logging.getLogger(__name__)


def retry(max_attempts: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """Retry a function on transient exceptions with exponential backoff.

    Only retries on OSError and RuntimeError (network/GPU transient failures).
    Does NOT retry on file-not-found or config errors.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except (OSError, RuntimeError) as e:
                    last_exception = e
                    if attempt < max_attempts:
                        wait = delay * (backoff ** (attempt - 1))
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}): "
                            f"{e}. Retrying in {wait:.1f}s..."
                        )
                        time.sleep(wait)
            raise last_exception
        return wrapper
    return decorator


def ensure_dir(path):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_filename(name):
    """Sanitize a string for use as a filename."""
    invalid_chars = '<>:"/\\|?*'
    for c in invalid_chars:
        name = name.replace(c, "_")
    return name[:100]


def format_duration(seconds):
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m"


def extract_guest_names(description):
    """Extract potential guest names from a text description."""
    # Pattern: common patterns for guest mentions
    patterns = [
        r"with\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        r"guest\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        r"featuring\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        r"and\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+discusses",
    ]
    names = []
    for pattern in patterns:
        matches = re.findall(pattern, description)
        names.extend(matches)
    return list(dict.fromkeys(names))  # deduplicate preserving order


def get_timestamp_suffix():
    """Generate a timestamp suffix for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_url(url):
    """Basic URL validation for YouTube URLs."""
    youtube_patterns = [
        r"https?://www\.youtube\.com/watch\?v=[\w-]+",
        r"https?://youtu\.be/[\w-]+",
        r"https?://www\.youtube\.com/playlist\?list=[\w-]+",
        r"https?://www\.youtube\.com/channel/[\w-]+",
        r"https?://www\.youtube\.com/@[\w-]+",
    ]
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False


def assign_speaker_labels(diarization_segments, metadata=None, existing_map=None):
    """Assign speaker labels to diarization segments.

    Strategy: first unique speaker = podcaster (from metadata.uploader or "Podcaster"),
    remaining speakers = Guest 1, Guest 2, etc.

    Args:
        diarization_segments: list of dicts with 'speaker' and 'start' keys
        metadata: object with optional 'uploader' attribute
        existing_map: dict to extend (for incremental chunk processing); if None, starts fresh

    Returns:
        Updated speaker label map {pyannote_id: label}
    """
    labels = existing_map.copy() if existing_map is not None else {}
    sorted_segs = sorted(diarization_segments, key=lambda s: s["start"])

    for seg in sorted_segs:
        pyannote_id = seg["speaker"]
        if pyannote_id in labels:
            continue  # already assigned (from previous chunks)

        if not labels:  # first speaker globally or in this batch
            if metadata and hasattr(metadata, "uploader") and metadata.uploader:
                labels[pyannote_id] = metadata.uploader
            else:
                labels[pyannote_id] = "Podcaster"
        else:
            guest_num = len([v for v in labels.values() if v.startswith("Guest ")]) + 1
            labels[pyannote_id] = f"Guest {guest_num}"

    return labels
