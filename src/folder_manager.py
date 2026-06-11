"""Folder naming and management utilities for PodAgent."""

import os
import re
from datetime import datetime


def generate_output_folder_name(title: str) -> str:
    """Generate a structured output folder name: output_<date>_<shortened_title>.

    Format: output_YYYYMMDD_<shortened_title>
    - Date is the current date
    - Title is sanitized, stripped of tags/hashtags, truncated to ~60 chars at word boundary
    """
    today = datetime.now().strftime("%Y%m%d")

    # Strip hashtags and extra tags from title
    cleaned = re.sub(r'#\w+', '', title)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Sanitize for filesystem — ASCII invalid chars + Unicode curly/smart quotes
    invalid_chars = '<>:"/\\|?*'
    # U+201C LEFT DOUBLE QUOTATION MARK, U+201D RIGHT DOUBLE QUOTATION MARK
    # U+2018 LEFT SINGLE QUOTATION MARK,   U+2019 RIGHT SINGLE QUOTATION MARK
    unicode_quotes = set('\u201c\u201d\u2018\u2019')
    for c in invalid_chars:
        cleaned = cleaned.replace(c, "_")
    cleaned = ''.join('_' if ch in unicode_quotes else ch for ch in cleaned)

    # Truncate to ~60 chars at a word boundary
    if len(cleaned) > 60:
        truncated = cleaned[:60]
        # Back up to the last space so we don't cut mid-word
        last_space = truncated.rfind(' ')
        if last_space > 30:  # keep at least ~30 chars before truncation
            truncated = truncated[:last_space]
        cleaned = truncated.rstrip()

    return f"output_{today}_{cleaned}"


def get_video_folder(base_data_dir: str, title: str) -> str:
    """Get or create the per-video data folder using structured naming."""
    folder_name = generate_output_folder_name(title)
    folder = os.path.join(base_data_dir, folder_name)
    os.makedirs(folder, exist_ok=True)
    return folder


def get_subfolder(parent: str, sub: str) -> str:
    """Get or create a subfolder (audio, transcript, analysis)."""
    path = os.path.join(parent, sub)
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    invalid_chars = '<>:"/\\|?*'
    # U+201C LEFT DOUBLE QUOTATION MARK, U+201D RIGHT DOUBLE QUOTATION MARK
    # U+2018 LEFT SINGLE QUOTATION MARK,   U+2019 RIGHT SINGLE QUOTATION MARK
    unicode_quotes = set('\u201c\u201d\u2018\u2019')
    for c in invalid_chars:
        name = name.replace(c, "_")
    name = ''.join('_' if ch in unicode_quotes else ch for ch in name)
    name = name.strip()
    return name[:80]
