"""Shared utilities for PodAgent."""

import os
import logging


def ensure_dir(path):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_filename(name):
    """Sanitize a string for use as a filename."""
    invalid_chars = '<>:"/\\|?*'
    for c in invalid_chars:
        name = name.replace(c, "_")
    name = name.strip()
    return name[:100]  # Limit length


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
