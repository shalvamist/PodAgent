import pytest
import tempfile
import os
from src.transcriber import WhisperTranscriber


def test_transcriber_initialization():
    transcriber = WhisperTranscriber(model="turbo")
    assert transcriber.model_name == "turbo"
    assert transcriber.language is None
    assert transcriber.beam_size == 5
    assert transcriber.carry_initial_prompt is False


def test_transcriber_custom_config():
    transcriber = WhisperTranscriber(
        model="medium",
        language="en",
        beam_size=10,
        initial_prompt="tech podcast about AI",
        carry_initial_prompt=True
    )
    assert transcriber.model_name == "medium"
    assert transcriber.language == "en"
    assert transcriber.beam_size == 10
    assert transcriber.initial_prompt == "tech podcast about AI"
    assert transcriber.carry_initial_prompt is True


def test_transcription_file_not_found():
    transcriber = WhisperTranscriber(model="medium")
    result = transcriber.transcribe("/nonexistent/path/audio.mp3")
    assert result.success is False
    assert result.error is not None
    assert "File not found" in result.error


def test_transcription_result_dataclass():
    from src.transcriber import TranscriptionResult
    result = TranscriptionResult(
        audio_path="/tmp/test.mp3",
        text="Hello world",
        language="en",
        segments=[{"start": 0.0, "end": 1.5, "text": "Hello world"}],
        duration=1.5,
        success=True,
        error=None
    )
    assert result.success is True
    assert result.text == "Hello world"
    assert result.language == "en"


def test_context_prompt_building():
    transcriber = WhisperTranscriber()
    video_info = {
        "title": "AI Podcast with John Doe and Jane Smith",
        "description": "Join us with John Doe and Jane Smith discussing machine learning",
        "channel": "TechTalk",
        "tags": ["AI", "machine learning", "podcast"],
        "uploader": "TechTalk Host",
    }
    prompt = transcriber.build_context_prompt(video_info)
    assert prompt is not None
    assert len(prompt) > 0
    # Should contain guest names extracted from description
    assert "John Doe" in prompt or "Jane Smith" in prompt


def test_context_prompt_empty_info():
    transcriber = WhisperTranscriber()
    video_info = {}
    prompt = transcriber.build_context_prompt(video_info)
    assert prompt is None
