"""Shared utilities for PodAgent."""

import os
import logging
import re
from datetime import datetime


logger = logging.getLogger(__name__)


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
