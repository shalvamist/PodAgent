import pytest
import tempfile
import os
import json
from src.transcript_builder import TranscriptBuilder, SpeakerProfile, StructuredTranscript
from src.downloader import VideoMetadata


def test_transcript_builder_initialization():
    builder = TranscriptBuilder()
    assert builder is not None
    assert os.path.isdir(builder.transcript_dir)


def test_transcript_builder_custom_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = TranscriptBuilder(transcript_dir=tmpdir)
        assert builder.transcript_dir == tmpdir


def test_speaker_profile_dataclass():
    profile = SpeakerProfile(
        speaker_id="speaker_0",
        label="Podcaster",
        first_appearance=0.0
    )
    assert profile.speaker_id == "speaker_0"
    assert profile.label == "Podcaster"
    assert profile.first_appearance == 0.0


def test_structured_transcript_dataclass():
    transcript = StructuredTranscript(
        video_title="Test Podcast",
        audio_path="/tmp/test.mp3",
        language="en",
        duration=3600.0,
        speakers=[
            SpeakerProfile(speaker_id="speaker_0", label="Podcaster", first_appearance=0.0),
            SpeakerProfile(speaker_id="speaker_1", label="Guest 1", first_appearance=5.0),
        ],
        segments=[
            {"start": 0.0, "end": 5.0, "speaker_label": "Podcaster", "text": "Welcome"},
            {"start": 5.0, "end": 10.0, "speaker_label": "Guest 1", "text": "Hello"},
        ],
        raw_text="Welcome Hello",
        output_path="/tmp/test_transcript.json"
    )
    assert transcript.video_title == "Test Podcast"
    assert len(transcript.speakers) == 2
    assert len(transcript.segments) == 2


def test_extract_guest_names_from_metadata():
    builder = TranscriptBuilder()
    metadata = VideoMetadata(
        video_id="abc123",
        title="Podcast with John Doe and Jane Smith",
        description="Join us with John Doe and Jane Smith discussing AI",
        channel="TechTalk",
        channel_id="UC_test",
        uploader="TechTalk Host",
        upload_date="20260101",
        tags=["AI", "podcast"],
        categories=["Education"],
        duration=3600.0,
        view_count=10000,
        like_count=500,
        thumbnail_url="https://img.youtube.com/vi/abc123/maxresdefault.jpg"
    )
    guest_to_label = builder.extract_guest_names_from_metadata(metadata)
    assert guest_to_label is not None
    assert len(guest_to_label) > 0


def test_extract_guest_names_empty_description():
    builder = TranscriptBuilder()
    metadata = VideoMetadata(
        video_id="abc123",
        title="Podcast",
        description="",
        channel="TechTalk",
        channel_id="UC_test",
        uploader="TechTalk Host",
        upload_date="20260101",
        tags=[],
        categories=[],
        duration=3600.0,
        view_count=10000,
        like_count=500,
        thumbnail_url=None
    )
    guest_to_label = builder.extract_guest_names_from_metadata(metadata)
    assert guest_to_label == {}


def test_build_structured_transcript():
    builder = TranscriptBuilder(transcript_dir=tempfile.mkdtemp())

    # Mock transcription result
    class MockTranscription:
        audio_path = "/tmp/test.mp3"
        text = "Welcome to the podcast. Hello from our guest."
        language = "en"
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Welcome to the podcast."},
            {"start": 5.0, "end": 10.0, "text": "Hello from our guest."},
        ]
        duration = 10.0
        success = True

    # Mock diarization result
    class MockDiarization:
        audio_path = "/tmp/test.mp3"
        speaker_segments = [
            {"start": 0.0, "end": 5.0, "speaker": "speaker_0"},
            {"start": 5.0, "end": 10.0, "speaker": "speaker_1"},
        ]
        num_speakers = 2
        duration = 10.0
        success = True

    transcript = builder.build(
        MockTranscription(),
        MockDiarization(),
        video_title="Test Podcast",
        podcaster_speaker="speaker_0"
    )

    assert transcript.output_path is not None
    assert os.path.exists(transcript.output_path)
    assert len(transcript.segments) == 2

    # Verify JSON file content
    with open(transcript.output_path, "r") as f:
        data = json.load(f)
    assert data["video_title"] == "Test Podcast"
    assert data["language"] == "en"
    assert len(data["segments"]) == 2


def test_build_with_metadata_context():
    builder = TranscriptBuilder(transcript_dir=tempfile.mkdtemp())

    class MockTranscription:
        audio_path = "/tmp/test.mp3"
        text = "Welcome. Hello guest."
        language = "en"
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Welcome."},
            {"start": 5.0, "end": 10.0, "text": "Hello guest."},
        ]
        duration = 10.0
        success = True

    class MockDiarization:
        audio_path = "/tmp/test.mp3"
        speaker_segments = [
            {"start": 0.0, "end": 5.0, "speaker": "speaker_0"},
            {"start": 5.0, "end": 10.0, "speaker": "speaker_1"},
        ]
        num_speakers = 2
        duration = 10.0
        success = True

    metadata = VideoMetadata(
        video_id="abc123",
        title="Podcast with John Doe",
        description="Join us with John Doe discussing AI",
        channel="TechTalk",
        channel_id="UC_test",
        uploader="TechTalk Host",
        upload_date="20260101",
        tags=["AI"],
        categories=["Education"],
        duration=3600.0,
        view_count=10000,
        like_count=500,
        thumbnail_url=None
    )

    transcript = builder.build(
        MockTranscription(),
        MockDiarization(),
        video_title="Podcast with John Doe",
        metadata=metadata,
        podcaster_speaker="speaker_0"
    )

    assert transcript.output_path is not None
    assert os.path.exists(transcript.output_path)
