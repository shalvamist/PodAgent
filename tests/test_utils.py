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


# --- Retry decorator tests ---

def test_retry_succeeds_on_first_try():
    from src.utils import retry
    call_count = 0

    @retry(max_attempts=3, delay=0.01)
    def always_ok():
        nonlocal call_count
        call_count += 1
        return "ok"

    assert always_ok() == "ok"
    assert call_count == 1


def test_retry_succeeds_after_failures():
    from src.utils import retry
    call_count = 0

    @retry(max_attempts=3, delay=0.01)
    def fails_twice_then_ok():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise OSError("transient")
        return "ok"

    assert fails_twice_then_ok() == "ok"
    assert call_count == 3


def test_retry_raises_after_max_attempts():
    from src.utils import retry
    call_count = 0

    @retry(max_attempts=3, delay=0.01)
    def always_fails():
        nonlocal call_count
        call_count += 1
        raise OSError("transient")

    with pytest.raises(OSError, match="transient"):
        always_fails()
    assert call_count == 3


def test_retry_does_not_retry_value_errors():
    from src.utils import retry
    call_count = 0

    @retry(max_attempts=3, delay=0.01)
    def raises_value_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        raises_value_error()
    assert call_count == 1  # No retry on ValueError


def test_retry_preserves_function_metadata():
    from src.utils import retry

    @retry(max_attempts=3, delay=0.01)
    def my_func(x, y=10):
        """My docstring."""
        return x + y

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring."


# --- Speaker label assignment tests ---

def test_assign_speaker_labels_first_is_podcaster():
    from src.utils import assign_speaker_labels

    segments = [
        {"speaker": "SPEAKER_01", "start": 0, "end": 10},
        {"speaker": "SPEAKER_02", "start": 15, "end": 25},
    ]

    labels = assign_speaker_labels(segments)
    assert labels["SPEAKER_01"] == "Podcaster"
    assert labels["SPEAKER_02"] == "Guest 1"


def test_assign_speaker_labels_with_metadata_uploader():
    from src.utils import assign_speaker_labels

    segments = [
        {"speaker": "SPEAKER_01", "start": 0, "end": 10},
        {"speaker": "SPEAKER_02", "start": 15, "end": 25},
    ]

    class FakeMetadata:
        uploader = "Danny Jones Clips"

    labels = assign_speaker_labels(segments, metadata=FakeMetadata())
    assert labels["SPEAKER_01"] == "Danny Jones Clips"
    assert labels["SPEAKER_02"] == "Guest 1"


def test_assign_speaker_labels_multiple_guests():
    from src.utils import assign_speaker_labels

    segments = [
        {"speaker": "SPEAKER_01", "start": 0, "end": 5},
        {"speaker": "SPEAKER_02", "start": 10, "end": 15},
        {"speaker": "SPEAKER_03", "start": 20, "end": 25},
    ]

    labels = assign_speaker_labels(segments)
    assert labels["SPEAKER_01"] == "Podcaster"
    assert labels["SPEAKER_02"] == "Guest 1"
    assert labels["SPEAKER_03"] == "Guest 2"


def test_assign_speaker_labels_empty_segments():
    from src.utils import assign_speaker_labels

    labels = assign_speaker_labels([])
    assert labels == {}


def test_assign_speaker_labels_single_speaker():
    from src.utils import assign_speaker_labels

    segments = [
        {"speaker": "SPEAKER_01", "start": 0, "end": 60},
    ]

    labels = assign_speaker_labels(segments)
    assert labels["SPEAKER_01"] == "Podcaster"


def test_assign_speaker_labels_reuses_existing_map():
    from src.utils import assign_speaker_labels

    segments_a = [
        {"speaker": "SPEAKER_01", "start": 0, "end": 10},
    ]
    labels_a = assign_speaker_labels(segments_a)
    assert labels_a["SPEAKER_01"] == "Podcaster"

    # Second chunk with a new speaker should extend the map
    segments_b = [
        {"speaker": "SPEAKER_02", "start": 0, "end": 10},
    ]
    labels_b = assign_speaker_labels(segments_b, existing_map=labels_a)
    assert labels_b["SPEAKER_01"] == "Podcaster"
    assert labels_b["SPEAKER_02"] == "Guest 1"
