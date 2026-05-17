import pytest
import tempfile
import os
import json
from src.downloader import YouTubeAudioDownloader, VideoMetadata, AudioDownloadResult
from src.transcriber import WhisperTranscriber, TranscriptionResult
from src.diarizer import SpeakerDiarizer, DiarizationResult
from src.transcript_builder import TranscriptBuilder, StructuredTranscript, SpeakerProfile
from src.storage import PodcastStorage


@pytest.mark.integration
def test_pipeline_integration_mock():
    """Test the full pipeline with mock data to verify integration between modules."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = os.path.join(tmpdir, "audio")
        transcript_dir = os.path.join(tmpdir, "transcripts")
        db_path = os.path.join(tmpdir, "podagent.db")

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(transcript_dir, exist_ok=True)

        # Create mock audio file
        audio_path = os.path.join(audio_dir, "test_audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(b"\x00" * 100)  # Dummy audio bytes

        # Create mock video metadata (all required fields)
        metadata = VideoMetadata(
            video_id="test_video_1",
            title="Test Podcast Integration",
            description="Guest: John Doe Jane Smith",
            channel="Test Channel",
            channel_id="UC_test",
            uploader="Test Uploader",
            upload_date="20260516",
            tags=["AI", "machine learning", "podcast"],
            categories=["Education"],
            duration=3600.0,
            view_count=1000,
            like_count=100,
            thumbnail_url="https://example.com/thumb.jpg",
        )

        # Create mock download result
        download_result = AudioDownloadResult(
            video_id="test_video_1",
            title="Test Podcast Integration",
            audio_path=audio_path,
            metadata=metadata,
            duration=3600.0,
            success=True,
            error=None,
        )

        # Create mock transcription result
        transcription_result = TranscriptionResult(
            audio_path=audio_path,
            text="Welcome to the podcast. Hello from the guest.",
            segments=[
                {"start": 0.0, "end": 5.0, "speaker": "speaker_0", "text": "Welcome to the podcast."},
                {"start": 5.0, "end": 10.0, "speaker": "speaker_1", "text": "Hello from the guest."},
            ],
            language="en",
            duration=3600.0,
            success=True,
            error=None,
        )

        # Create mock diarization result
        diarization_result = DiarizationResult(
            audio_path=audio_path,
            speaker_segments=[
                {"start": 0.0, "end": 5.0, "speaker": "speaker_0"},
                {"start": 5.0, "end": 10.0, "speaker": "speaker_1"},
            ],
            num_speakers=2,
            duration=3600.0,
            success=True,
            error=None,
        )

        # Initialize pipeline components
        downloader = YouTubeAudioDownloader(
            audio_format="mp3",
            output_dir=audio_dir,
        )
        transcriber = WhisperTranscriber(
            model="turbo",
            language="en",
            carry_initial_prompt=True,
        )
        diarizer = SpeakerDiarizer(
            hf_token="test_token",
        )
        builder = TranscriptBuilder(
            transcript_dir=transcript_dir,
        )
        storage = PodcastStorage(
            db_path=db_path,
        )

        # Step 1: Download (mock)
        assert download_result.success
        assert download_result.audio_path == audio_path
        assert download_result.metadata.video_id == "test_video_1"

        # Step 2: Transcribe (mock)
        assert transcription_result.success
        assert len(transcription_result.text) > 0
        assert transcription_result.language == "en"

        # Step 3: Diarize (mock)
        assert diarization_result.success
        assert diarization_result.num_speakers == 2
        assert len(diarization_result.speaker_segments) == 2

        # Step 4: Build transcript
        transcript = builder.build(
            transcription_result,
            diarization_result,
            video_title=metadata.title,
            metadata=metadata,
            podcaster_speaker=None,
        )

        assert transcript is not None
        assert transcript.video_title == "Test Podcast Integration"
        assert len(transcript.segments) > 0
        assert len(transcript.speakers) == 2
        assert os.path.isfile(transcript.output_path)

        # Step 5: Save to storage
        storage.save_podcast({
            "video_id": metadata.video_id,
            "title": metadata.title,
            "channel_id": metadata.channel_id,
            "channel_name": metadata.channel,
            "audio_path": audio_path,
            "transcript_path": transcript.output_path,
            "language": transcription_result.language,
            "duration": transcription_result.duration,
            "num_speakers": diarization_result.num_speakers,
        })

        storage.save_segments(1, transcript.segments)
        storage.save_speakers(1, transcript.speakers)

        # Verify storage
        all_podcasts = storage.get_all_podcasts()
        assert len(all_podcasts) == 1
        assert all_podcasts[0]["video_id"] == "test_video_1"
        assert all_podcasts[0]["title"] == "Test Podcast Integration"

        # Verify transcript file content
        with open(transcript.output_path, "r") as f:
            transcript_data = json.load(f)
        assert "segments" in transcript_data
        assert "speakers" in transcript_data
        assert "video_title" in transcript_data
        assert len(transcript_data["segments"]) > 0


@pytest.mark.integration
def test_pipeline_failure_handling():
    """Test that the pipeline handles failures gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = os.path.join(tmpdir, "audio")
        transcript_dir = os.path.join(tmpdir, "transcripts")
        db_path = os.path.join(tmpdir, "podagent.db")

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(transcript_dir, exist_ok=True)

        # Create mock download result with failure
        download_result = AudioDownloadResult(
            video_id="",
            title="",
            audio_path="",
            metadata=VideoMetadata(
                video_id="",
                title="",
                description="",
                channel="",
                channel_id="",
                uploader="",
                upload_date="",
                tags=[],
                categories=[],
                duration=None,
                view_count=None,
                like_count=None,
                thumbnail_url=None,
            ),
            duration=None,
            success=False,
            error="Download failed: video not found",
        )

        assert not download_result.success
        assert download_result.error is not None

        # Transcription should not proceed without audio
        transcriber = WhisperTranscriber()
        transcription_result = transcriber.transcribe("/nonexistent/audio.mp3")
        assert not transcription_result.success
        assert transcription_result.error is not None

        # Diarization should not proceed without audio
        diarizer = SpeakerDiarizer(hf_token="test_token")
        diarization_result = diarizer.diarize("/nonexistent/audio.mp3")
        assert not diarization_result.success
        assert diarization_result.error is not None
