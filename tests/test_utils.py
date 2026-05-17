import pytest
from src.utils import (
    ensure_dir,
    sanitize_filename,
    format_duration,
    extract_guest_names,
    get_timestamp_suffix,
    validate_url,
)


def test_ensure_dir():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        sub = f"{tmpdir}/subdir/nested"
        result = ensure_dir(sub)
        assert result == sub
        assert __import__("os").path.isdir(sub)


def test_sanitize_filename():
    assert sanitize_filename("Hello: World/") == "Hello_ World_"
    assert sanitize_filename("A" * 200) == "A" * 100
    assert sanitize_filename("normal_name") == "normal_name"


def test_format_duration():
    assert format_duration(30) == "30s"
    assert format_duration(90) == "1m 30s"
    assert format_duration(3661) == "1h 1m"


def test_extract_guest_names():
    desc = "Tech Talk with John Doe and Jane Smith discussing AI"
    names = extract_guest_names(desc)
    assert "John Doe" in names
    assert "Jane Smith" in names


def test_extract_guest_names_empty():
    names = extract_guest_names("no names here")
    assert names == []


def test_get_timestamp_suffix():
    suffix = get_timestamp_suffix()
    assert len(suffix) == 15  # YYYYMMDD_HHMMSS


def test_validate_url():
    assert validate_url("https://www.youtube.com/watch?v=abc123") is True
    assert validate_url("https://youtu.be/abc123") is True
    assert validate_url("https://www.youtube.com/playlist?list=abc123") is True
    assert validate_url("https://example.com/random") is False
