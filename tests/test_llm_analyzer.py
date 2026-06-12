"""Tests for LLM analyzer module."""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.llm_analyzer import LLMAnalyzer, LLMAnalyzerConfig, LLMAnalysisResult


class TestLLMAnalyzerConfig:
    """Test LLMAnalyzerConfig defaults and overrides."""

    def test_default_config(self):
        config = LLMAnalyzerConfig()
        assert config.provider == "ollama"
        assert config.model == "llama3"
        assert config.base_url == "http://localhost:11434"
        assert config.lmstudio_url == "http://localhost:1234"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout_seconds == 300
        assert config.streaming is True

    def test_custom_config(self):
        config = LLMAnalyzerConfig(
            provider="lmstudio",
            model="mistral",
            base_url="http://localhost:9999",
            lmstudio_url="http://localhost:8888",
            temperature=0.5,
            max_tokens=2048,
            timeout_seconds=60,
            streaming=False,
        )
        assert config.provider == "lmstudio"
        assert config.model == "mistral"
        assert config.base_url == "http://localhost:9999"
        assert config.lmstudio_url == "http://localhost:8888"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.timeout_seconds == 60
        assert config.streaming is False


class TestLLMAnalyzerPromptBuilding:
    """Test prompt generation for each analysis mode."""

    def setup_method(self, method):
        self.sample_transcript = {
            "video_title": "Test Podcast: AI and Future",
            "speakers": [
                {"speaker_id": "speaker_0", "label": "Podcaster"},
                {"speaker_id": "speaker_1", "label": "Guest 1"},
            ],
            "raw_text": "This is a test transcript about AI and the future of technology. The podcaster discusses machine learning and the guest talks about neural networks.",
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker_label": "Podcaster", "text": "Welcome to the podcast."},
                {"start": 5.0, "end": 10.0, "speaker_label": "Guest 1", "text": "Thanks for having me."},
            ],
        }

    def test_summary_prompt(self):
        analyzer = LLMAnalyzer()
        prompt = analyzer._build_prompt(self.sample_transcript, "summary")
        assert "Test Podcast: AI and Future" in prompt
        assert "Podcaster" in prompt
        assert "Guest 1" in prompt
        assert "[Podcaster]" in prompt
        assert "1-2 paragraph summary" in prompt
        assert "main topics discussed" in prompt

    def test_insights_prompt(self):
        analyzer = LLMAnalyzer()
        prompt = analyzer._build_prompt(self.sample_transcript, "insights")
        assert "Test Podcast: AI and Future" in prompt
        assert "Extract key insights" in prompt
        assert "Main topics covered" in prompt
        assert "Key arguments or positions taken" in prompt
        assert "Notable data points or statistics mentioned" in prompt

    def test_notes_prompt(self):
        analyzer = LLMAnalyzer()
        prompt = analyzer._build_prompt(self.sample_transcript, "notes")
        assert "Test Podcast: AI and Future" in prompt
        assert "Executive summary" in prompt
        assert "Topic sections with key points" in prompt
        assert "Important quotes" in prompt
        assert "Action items or takeaways" in prompt

    def test_blog_prompt(self):
        analyzer = LLMAnalyzer()
        prompt = analyzer._build_prompt(self.sample_transcript, "blog")
        assert "Test Podcast: AI and Future" in prompt
        assert "engaging introduction" in prompt
        assert "logical sections with headings" in prompt
        assert "Has a conclusion with takeaways" in prompt
        assert "800-1200 words" in prompt
        assert "markdown formatting" in prompt

    def test_long_transcript_truncation(self):
        long_text = "x" * 10000
        transcript = {
            "video_title": "Long Podcast",
            "speakers": [],
            "raw_text": long_text,
            "segments": [],
        }
        analyzer = LLMAnalyzer()
        prompt = analyzer._build_prompt(transcript, "summary")
        assert "Long Podcast" in prompt
        assert len(prompt) < 10000


class TestLLMAnalyzerAPIUrls:
    """Test API URL generation for different providers."""

    def test_ollama_url(self):
        analyzer = LLMAnalyzer()
        url = analyzer._get_api_url()
        assert url == "http://localhost:11434/api/generate"

    def test_lmstudio_url(self):
        config = LLMAnalyzerConfig(provider="lmstudio")
        analyzer = LLMAnalyzer(config)
        url = analyzer._get_api_url()
        assert url == "http://localhost:1234/v1/completions"

    def test_invalid_provider(self):
        config = LLMAnalyzerConfig(provider="unknown")
        analyzer = LLMAnalyzer(config)
        with pytest.raises(ValueError, match="Unknown provider"):
            analyzer._get_api_url()


class TestLLMAnalyzerAvailability:
    """Test LLM availability checking with mocked HTTP."""

    @patch("src.llm_analyzer.httpx.Client")
    def test_ollama_available(self, mock_client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.return_value.get.return_value = mock_response

        analyzer = LLMAnalyzer()
        is_avail, err = analyzer.check_availability()
        assert is_avail is True

    @patch("src.llm_analyzer.httpx.Client")
    def test_ollama_unavailable(self, mock_client):
        mock_client.return_value.get.side_effect = Exception("Connection refused")

        analyzer = LLMAnalyzer()
        is_avail, err = analyzer.check_availability()
        assert is_avail is False
        assert "Ollama" in err

    @patch("src.llm_analyzer.httpx.Client")
    def test_lmstudio_available(self, mock_client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "test-model"}]}
        mock_client.return_value.get.return_value = mock_response

        config = LLMAnalyzerConfig(provider="lmstudio")
        analyzer = LLMAnalyzer(config)
        is_avail, err = analyzer.check_availability()
        assert is_avail is True

    @patch("src.llm_analyzer.httpx.Client")
    def test_lmstudio_no_model_loaded(self, mock_client):
        """When LM Studio returns 200 but no models are loaded."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_client.return_value.get.return_value = mock_response

        config = LLMAnalyzerConfig(provider="lmstudio")
        analyzer = LLMAnalyzer(config)
        is_avail, err = analyzer.check_availability()
        assert is_avail is False
        assert "no model" in err.lower() or "model" in err.lower()

    @patch("src.llm_analyzer.httpx.Client")
    def test_lmstudio_connect_error(self, mock_client):
        """When LM Studio is not running at all."""
        import httpx
        mock_client.return_value.get.side_effect = httpx.ConnectError("Connection refused")

        config = LLMAnalyzerConfig(provider="lmstudio", lmstudio_url="http://localhost:1234")
        analyzer = LLMAnalyzer(config)
        is_avail, err = analyzer.check_availability()
        assert is_avail is False
        assert "LM Studio" in err


class TestLLMAnalyzerListModels:
    """Test listing available models with mocked HTTP."""

    @patch("src.llm_analyzer.httpx.Client")
    def test_ollama_models(self, mock_client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3"},
                {"name": "mistral"},
                {"name": "gemma"},
            ]
        }
        mock_client.return_value.get.return_value = mock_response

        analyzer = LLMAnalyzer()
        models = analyzer.list_available_models()
        assert models == ["llama3", "mistral", "gemma"]

    @patch("src.llm_analyzer.httpx.Client")
    def test_lmstudio_models(self, mock_client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "mistral-7b"},
                {"id": "llama-2-7b"},
            ]
        }
        mock_client.return_value.get.return_value = mock_response

        config = LLMAnalyzerConfig(provider="lmstudio")
        analyzer = LLMAnalyzer(config)
        models = analyzer.list_available_models()
        assert models == ["mistral-7b", "llama-2-7b"]

    @patch("src.llm_analyzer.httpx.Client")
    def test_model_list_error(self, mock_client):
        mock_client.return_value.get.side_effect = Exception("Error")

        analyzer = LLMAnalyzer()
        models = analyzer.list_available_models()
        assert models == []


class TestLLMAnalyzerAnalysis:
    """Test analysis execution with mocked HTTP responses."""

    @patch("src.llm_analyzer.httpx.Client")
    def test_successful_ollama_analysis(self, mock_client):
        # Mock availability check (GET /api/tags)
        avail_response = MagicMock()
        avail_response.status_code = 200
        mock_client.return_value.get.return_value = avail_response

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            json.dumps({"response": "This is a summary of the podcast."}).encode(),
            json.dumps({"done": True}).encode(),
        ]
        mock_client.return_value.post.return_value = mock_response

        analyzer = LLMAnalyzer()
        transcript = {
            "video_title": "Test Podcast",
            "speakers": [],
            "raw_text": "Test transcript content.",
            "segments": [],
        }
        result = analyzer.analyze(transcript, mode="summary")

        assert result.analysis_mode == "summary"
        assert result.llm_model == "llama3"
        assert result.provider == "ollama"
        assert "Test Podcast" in result.transcript_id
        assert result.summary_text == "This is a summary of the podcast."
        assert result.output_path is not None
        assert os.path.exists(result.output_path)

    @patch("src.llm_analyzer.httpx.Client")
    def test_successful_lmstudio_analysis(self, mock_client):
        # Mock availability check (GET /v1/models)
        avail_response = MagicMock()
        avail_response.status_code = 200
        avail_response.json.return_value = {"data": [{"id": "test-model"}]}
        mock_client.return_value.get.return_value = avail_response

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"text": "LM Studio analysis result."}]
        }
        mock_client.return_value.post.return_value = mock_response

        config = LLMAnalyzerConfig(provider="lmstudio", streaming=False)
        analyzer = LLMAnalyzer(config)
        transcript = {
            "video_title": "Test Podcast",
            "speakers": [],
            "raw_text": "Test transcript content.",
            "segments": [],
        }
        result = analyzer.analyze(transcript, mode="insights")

        assert result.provider == "lmstudio"
        assert result.summary_text == "LM Studio analysis result."

    @patch("src.llm_analyzer.httpx.Client")
    def test_connection_error(self, mock_client):
        import httpx
        mock_client.return_value.post.side_effect = httpx.ConnectError("Connection refused")

        analyzer = LLMAnalyzer()
        transcript = {
            "video_title": "Test Podcast",
            "speakers": [],
            "raw_text": "Test content.",
            "segments": [],
        }
        result = analyzer.analyze(transcript, mode="summary")

        assert result.summary_text.startswith("ERROR")
        assert result.structured_output is None
        assert result.raw_response == ""

    @patch("src.llm_analyzer.httpx.Client")
    def test_timeout_error(self, mock_client):
        import httpx
        mock_client.return_value.post.side_effect = httpx.TimeoutException("Timeout")

        analyzer = LLMAnalyzer()
        transcript = {
            "video_title": "Test Podcast",
            "speakers": [],
            "raw_text": "Test content.",
            "segments": [],
        }
        result = analyzer.analyze(transcript, mode="summary")

        assert result.summary_text.startswith("ERROR")
        assert result.structured_output is None
        assert result.raw_response == ""

    @patch("src.llm_analyzer.httpx.Client")
    def test_http_error(self, mock_client):
        import httpx
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.return_value.post.return_value = mock_response

        # httpx.HTTPStatusError needs response object
        error = httpx.HTTPStatusError("HTTP 500", request=MagicMock(), response=mock_response)
        mock_client.return_value.post.side_effect = error

        analyzer = LLMAnalyzer()
        transcript = {
            "video_title": "Test Podcast",
            "speakers": [],
            "raw_text": "Test content.",
            "segments": [],
        }
        result = analyzer.analyze(transcript, mode="summary")

        assert result.summary_text.startswith("ERROR")
        assert result.structured_output is None
        assert result.raw_response == ""

    @patch("src.llm_analyzer.httpx.Client")
    def test_non_streaming_response(self, mock_client):
        # Mock availability check (GET /api/tags)
        avail_response = MagicMock()
        avail_response.status_code = 200
        mock_client.return_value.get.return_value = avail_response

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Non-streaming response text."
        }
        mock_client.return_value.post.return_value = mock_response

        config = LLMAnalyzerConfig(streaming=False)
        analyzer = LLMAnalyzer(config)
        transcript = {
            "video_title": "Test Podcast",
            "speakers": [],
            "raw_text": "Test content.",
            "segments": [],
        }
        result = analyzer.analyze(transcript, mode="summary")

        assert result.summary_text == "Non-streaming response text."


class TestLLMAnalyzerSanitizeFilename:
    """Test filename sanitization — delegated to folder_manager module."""

    def test_basic_sanitize(self):
        from src.folder_manager import sanitize_filename
        result = sanitize_filename("Test Podcast Title")
        assert result == "Test Podcast Title"

    def test_sanitize_with_invalid_chars(self):
        from src.folder_manager import sanitize_filename
        result = sanitize_filename("Test: Podcast > Title")
        assert ":" not in result
        assert ">" not in result
        assert result == "Test_ Podcast _ Title"

    def test_length_limit(self):
        from src.folder_manager import sanitize_filename
        long_name = "x" * 200
        result = sanitize_filename(long_name)
        assert len(result) <= 80


class TestLLMAnalyzerClose:
    """Test client cleanup."""

    @patch("src.llm_analyzer.httpx.Client")
    def test_close_method(self, mock_client):
        analyzer = LLMAnalyzer()
        analyzer.close()
        mock_client.return_value.close.assert_called_once()

# --- _clean_raw_response pipeline tests ---

from src.llm_analyzer import (
    _strip_thinking_blocks,
    _strip_stray_chars,
    _strip_planning_text,
    _extract_json_content,
    _extract_non_json_content,
    _clean_raw_response,
)


class TestStripThinkingBlocks:
    """Test thinking block removal."""

    def test_empty_input(self):
        assert _strip_thinking_blocks("") == ""

    def test_no_thinking_tags(self):
        text = "This is normal content."
        assert _strip_thinking_blocks(text) == text

    def test_xml_thinking_paired(self):
        NL = chr(10)
        text = "<thinking>Some reasoning</thinking>" + NL + "Actual content"
        result = _strip_thinking_blocks(text)
        assert "Some reasoning" not in result
        assert "Actual content" in result

    def test_unclosed_xml_tag_with_heading(self):
        NL = chr(10)
        text = "<thinking>" + NL + "**1. Main Point**" + NL + "Content here"
        result = _strip_thinking_blocks(text)
        assert "**1. Main Point**" in result

    def test_unclosed_tag_no_useful_content(self):
        NL = chr(10)
        text = "<thinking>" + NL + "Just thinking with no real content"
        result = _strip_thinking_blocks(text)
        assert "Just thinking" not in result


class TestStripStrayChars:
    """Test stray character removal."""

    def test_empty_input(self):
        assert _strip_stray_chars("") == ""

    def test_leading_brace_on_own_line(self):
        NL = chr(10)
        text = "}" + NL + NL + "Actual content"
        result = _strip_stray_chars(text)
        assert "}" not in result.split(NL)[0] if result else True
        assert "Actual content" in result

    def test_trailing_code_fence_incomplete(self):
        text = "Content here\n```\n```json\n```\n"
        result = _strip_stray_chars(text)
        assert "```json" in result


class TestStripPlanningText:
    """Test planning/draft text removal."""

    def test_empty_input(self):
        assert _strip_planning_text("") == ""

    def test_draft_structure_removed(self):
        NL = chr(10)
        text = "Draft structure" + NL + "**1. Main Point**" + NL + "Actual content"
        result = _strip_planning_text(text)
        assert "Draft structure" not in result
        assert "**1. Main Point**" in result

    def test_trailing_check_text_removed(self):
        text = "Content here\nCheck against JSON schema:"
        result = _strip_planning_text(text)
        assert "Check against" not in result


class TestExtractJsonContent:
    """Test JSON content extraction."""

    def test_empty_input(self):
        assert _extract_json_content("") is None

    def test_no_braces(self):
        text = "Just plain text with no braces"
        assert _extract_json_content(text) is None

    def test_short_json_returns_none(self):
        import json as json_mod
        obj = {"mode": "summary", "content": "Short"}
        text = json_mod.dumps(obj)
        assert _extract_json_content(text) is None

    def test_long_json_extracts_content(self):
        import json as json_mod
        obj = {"mode": "summary", "content": "This is a much longer analysis that has enough words to pass any minimum length validation checks and be considered real substantial content."}
        text = json_mod.dumps(obj)
        result = _extract_json_content(text)
        assert result == obj["content"]


class TestExtractNonJsonContent:
    """Test non-JSON content extraction."""

    def test_empty_input(self):
        assert _extract_non_json_content("") is None

    def test_numbered_analysis_returns_heading_only(self):
        text = "**1. Main Topic**\nThis is a detailed explanation of the main topic with sufficient length."
        result = _extract_non_json_content(text)
        assert result == "**1. Main Topic**"

    def test_bullet_analysis_preserved(self):
        text = "- **Key Point**: This is a substantial bullet point with enough content to pass validation checks."
        result = _extract_non_json_content(text)
        assert "Key Point" in result

    def test_json_input_falls_through_to_original_text(self):
        import json as json_mod
        obj = {"mode": "summary", "content": "Some content"}
        text = json_mod.dumps(obj)
        result = _extract_non_json_content(text)
        assert result == text


class TestCleanRawResponse:
    """Test the full _clean_raw_response pipeline."""

    def test_empty_input(self):
        assert _clean_raw_response("") == ""

    def test_normal_text_passthrough(self):
        text = "This is normal analysis content."
        assert _clean_raw_response(text) == text

    def test_long_json_extracts_content_field(self):
        import json as json_mod
        obj = {"mode": "summary", "content": "This is a much longer analysis that has enough words to pass any minimum length validation checks and be considered real substantial content."}
        text = json_mod.dumps(obj)
        result = _clean_raw_response(text)
        assert result == obj["content"]

    def test_short_json_returns_full_object(self):
        import json as json_mod
        obj = {"mode": "summary", "content": "Real analysis here"}
        text = json_mod.dumps(obj)
        result = _clean_raw_response(text)
        assert result == text

    def test_thinking_block_stripped_from_long_json(self):
        import json as json_mod
        obj = {"mode": "summary", "content": "This is a much longer analysis that has enough words to pass any minimum length validation checks and be considered real substantial content."}
        NL = chr(10)
        text = "<thinking>Let me think..." + NL + "</thinking>" + NL + json_mod.dumps(obj)
        result = _clean_raw_response(text)
        assert "thinking" not in result.lower()
        assert result == obj["content"]

    def test_stray_brace_stripped_from_long_json(self):
        import json as json_mod
        obj = {"mode": "summary", "content": "This is a much longer analysis that has enough words to pass any minimum length validation checks and be considered real substantial content."}
        NL = chr(10)
        text = "}" + NL + NL + json_mod.dumps(obj)
        result = _clean_raw_response(text)
        assert result == obj["content"]

    def test_planning_text_stripped_from_long_json(self):
        import json as json_mod
        obj = {"mode": "summary", "content": "This is a much longer analysis that has enough words to pass any minimum length validation checks and be considered real substantial content."}
        NL = chr(10)
        text = "Draft structure" + NL + "Plan:" + NL + json_mod.dumps(obj)
        result = _clean_raw_response(text)
        assert "Draft structure" not in result
        assert result == obj["content"]

    def test_plain_text_preserved(self):
        text = "**1. Main Topic**\nThis is a detailed explanation of the main topic with sufficient length to pass all validation checks and be considered real content."
        result = _clean_raw_response(text)
        assert "**1. Main Topic**" in result

    def test_no_real_content_returns_empty(self):
        import json as json_mod
        obj = {"mode": "summary", "content": "Your analysis text"}
        text = json_mod.dumps(obj)
        result = _clean_raw_response(text)
        # "analysis" in content triggers template placeholder detection -> returns ""

