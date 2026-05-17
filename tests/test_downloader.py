import pytest
import tempfile
import os
from src.downloader import YouTubeAudioDownloader


def test_downloader_initialization():
    downloader = YouTubeAudioDownloader()
    assert downloader.audio_format == "mp3"
    assert downloader.audio_quality == "best"
    assert os.path.isdir(downloader.output_dir)


def test_downloader_custom_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = YouTubeAudioDownloader(
            audio_format="wav",
            audio_quality="5",
            output_dir=tmpdir
        )
        assert downloader.audio_format == "wav"
        assert downloader.audio_quality == "5"
        assert downloader.output_dir == tmpdir


def test_download_audio_requires_url():
    downloader = YouTubeAudioDownloader()
    result = downloader.download_audio("https://www.youtube.com/watch?v=invalid_test")
    # This will fail because the URL is invalid, but we should get a proper error result
    assert result.success is False
    assert result.error is not None


def test_video_metadata_dataclass():
    from src.downloader import VideoMetadata
    meta = VideoMetadata(
        video_id="abc123",
        title="Test Podcast",
        description="A test podcast",
        channel="Test Channel",
        channel_id="UC_test",
        uploader="Test Host",
        upload_date="20260101",
        tags=["podcast", "tech"],
        categories=["Education"],
        duration=3600.0,
        view_count=10000,
        like_count=500,
        thumbnail_url="https://img.youtube.com/vi/abc123/maxresdefault.jpg"
    )
    assert meta.video_id == "abc123"
    assert meta.title == "Test Podcast"
    assert meta.duration == 3600.0


def test_audio_download_result_dataclass():
    from src.downloader import AudioDownloadResult
    result = AudioDownloadResult(
        video_id="abc123",
        title="Test Podcast",
        audio_path="/tmp/test.mp3",
        metadata=None,
        duration=3600.0,
        success=True,
        error=None
    )
    assert result.success is True
    assert result.error is None
