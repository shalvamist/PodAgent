import pytest
import tempfile
import os
from src.diarizer import SpeakerDiarizer


def test_diarizer_initialization():
    diarizer = SpeakerDiarizer(hf_token="dummy_token")
    assert diarizer.hf_token == "dummy_token"
    assert diarizer.pipeline_name == "pyannote/speaker-diarization-community-1"


def test_diarizer_custom_pipeline():
    diarizer = SpeakerDiarizer(
        hf_token="test_token",
        pipeline_name="pyannote/speaker-diarization@2023/07/18"
    )
    assert diarizer.hf_token == "test_token"
    assert diarizer.pipeline_name == "pyannote/speaker-diarization@2023/07/18"


def test_diarize_file_not_found():
    diarizer = SpeakerDiarizer(hf_token="dummy_token")
    result = diarizer.diarize("/nonexistent/path/audio.wav")
    assert result.success is False
    assert result.error is not None
    assert "File not found" in result.error


def test_diarization_result_dataclass():
    from src.diarizer import DiarizationResult
    result = DiarizationResult(
        audio_path="/tmp/test.wav",
        speaker_segments=[
            {"start": 0.0, "end": 5.0, "speaker": "speaker_0"},
            {"start": 5.0, "end": 10.0, "speaker": "speaker_1"},
        ],
        num_speakers=2,
        duration=10.0,
        success=True,
        error=None
    )
    assert result.success is True
    assert result.num_speakers == 2
    assert len(result.speaker_segments) == 2


def test_diarizer_no_token():
    diarizer = SpeakerDiarizer(hf_token=None)
    assert diarizer.hf_token is None
