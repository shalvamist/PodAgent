# PodAgent

YouTube podcast auto-monitor, transcriber, and diarizer with optional LLM analysis.

## Features

- Auto-monitor YouTube channels for new podcast uploads
- Download audio using yt-dlp with metadata extraction
- Transcribe audio using Whisper turbo with context-enhanced prompting
- Speaker diarization using pyannote.audio
- Generate structured transcripts with podcaster/guest identification
- SQLite storage for podcasts, segments, speaker profiles, and LLM analyses
- CLI entry point for single video, batch processing, or LLM analysis
- Optional local LLM analysis (Ollama/LM Studio): summary, insights, notes, blog

## Architecture

```
YouTube Channel --> ChannelMonitor --> yt-dlp (audio + metadata)
                                      --> WhisperTranscriber (text + segments)
                                      --> SpeakerDiarizer (speaker labels)
                                      --> TranscriptBuilder (structured output)
                                      --> PodcastStorage (SQLite database)
                                      --> LLMAnalyzer (optional: summary/insights/notes/blog)
```

Three-stage pipeline with metadata context:
1. **Download** — yt-dlp extracts audio + metadata from YouTube
2. **Transcribe** — Whisper turbo converts audio to text with context-enhanced prompting (guest names from description, tags, channel topic)
3. **Diarize** — pyannote.audio identifies speakers with timestamped segments
4. **Build** — combines transcription + diarization into structured transcript with speaker labels
5. **Store** — saves to SQLite database
6. **Analyze** — (optional) feeds transcript to local LLM for summary/insights/notes/blog

## Setup

### Prerequisites

1. Install ffmpeg: `sudo apt install ffmpeg`
2. Install Python dependencies: `pip install -r requirements.txt`
3. Create HuggingFace token: https://hf.co/settings/tokens (read-only)
4. Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### Configuration

1. Add channels to `data/channels.yaml`:
   ```yaml
   channels:
     - id: UC_x5XG1OV2P6uZZ5FSM9Ttw  # Example: Google Developers
       name: Google Developers
   ```

2. Set your HF token in `config.yaml`:
   ```yaml
   settings:
     diarization:
       hf_token: YOUR_HF_TOKEN_HERE
   ```

3. Configure transcription model (auto-fallback if GPU < 6GB):
   ```yaml
   settings:
     transcription:
       model: turbo  # large-v3-turbo (~6GB VRAM)
   ```

4. Configure LLM analysis (optional):
   ```yaml
   settings:
     llm:
       provider: ollama  # "ollama" or "lmstudio"
       model: llama3     # Must be loaded in Ollama/LM Studio
       base_url: http://localhost:11434
       lmstudio_url: http://localhost:1234
       temperature: 0.7
       max_tokens: 4096
       timeout_seconds: 120
       streaming: true
       enable_structured_output: true
   ```

### GPU Strategy

- Primary: NVIDIA GPU via CUDA (torch.cuda.is_available())
- Fallback: CPU inference with fp32 mode (~2-3x slower)
- GPU < 6GB VRAM: auto-switches to medium model instead of turbo
- VRAM requirements: tiny=1GB, base=1GB, small=2GB, medium=4GB, large-v3=10GB, large-v3-turbo=6GB

## Usage

### Process a single video

```bash
python run.py --url https://www.youtube.com/watch?v=VIDEO_ID
```

### Monitor all configured channels

```bash
python run.py --monitor
```

### Custom config file

```bash
python run.py --config /path/to/config.yaml
```

### Help

```bash
python run.py --help
```

### Run LLM analysis on transcripts

Requires Ollama or LM Studio running locally.

```bash
# Analyze a single video with LLM analysis
python run.py --url https://www.youtube.com/watch?v=VIDEO_ID --analyze

# Monitor channels with LLM analysis enabled
python run.py --monitor --analyze

# List stored LLM analyses
python run.py --list-analyses

# List stored podcasts
python run.py --list-podcasts
```

LLM analysis runs 4 modes per transcript:
- **summary** — 1-2 paragraph summary of main topics
- **insights** — structured bullet points of key insights
- **notes** — structured notes with speaker attribution
- **blog** — formatted article from transcript

Results saved to `data/llm_analysis/` and `podagent.db` (llm_analysis table).

## Output Structure

```
PodAgent/
├── src/
│   ├── downloader.py        # yt-dlp wrapper for audio extraction
│   ├── transcriber.py       # Whisper transcription pipeline
│   ├── diarizer.py          # pyannote speaker diarization
│   ├── transcript_builder.py # Structured transcript generation
│   ├── channel_monitor.py   # Channel polling for new uploads
│   ├── storage.py           # SQLite/JSON storage layer
│   ├── llm_analyzer.py      # LLM analysis module
│   └── utils.py             # Shared utilities
├── data/
│   ├── audio/               # Downloaded audio files
│   ├── transcripts/         # Generated transcript JSON files
│   ├── llm_analysis/        # LLM analysis results
│   ├── channels.yaml        # User-provided channel list
│   └── podagent.db          # SQLite database
├── tests/                   # Unit and integration tests
├── run.py                   # Main CLI entry point
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## Storage Schema

### podcasts table
- id (autoincrement)
- video_id (unique)
- title, channel_id, channel_name
- audio_path, transcript_path
- language, duration, num_speakers
- processed_at (timestamp)

### transcript_segments table
- podcast_id (FK to podcasts)
- start_time, end_time, speaker_label, text

### speakers table
- podcast_id (FK to podcasts)
- speaker_id, label, first_appearance

### llm_analysis table
- podcast_id (FK to podcasts)
- analysis_mode (summary/insights/notes/blog)
- llm_model, provider
- summary_text, structured_output (JSON string)
- processing_time (seconds)
- created_at (timestamp)

## Context-Enhanced Transcription

Whisper `initial_prompt` accepts custom text for the first decode window. With `carry_initial_prompt=True`, it prepends to every subsequent window.

Context is built from YouTube metadata:
- Guest names (extracted from description using regex patterns)
- Channel topic
- Tags
- Title keywords

Example: "John Doe Jane Smith Channel: TechTalk tags: AI machine learning podcast"

This makes Whisper more likely to correctly predict proper nouns, guest names, and domain-specific vocabulary.

## LLM Analysis

Feeds structured transcripts to local LLMs via HTTP API:

### Supported providers
- **Ollama** — http://localhost:11434/api/generate
- **LM Studio** — http://localhost:1234/v1/completions

### Analysis modes
- `summary` — concise 1-2 paragraph summary
- `insights` — structured bullet points of key insights
- `notes` — structured notes with speaker attribution
- `blog` — formatted article from transcript

### Configuration
- provider, model, base_url, temperature, max_tokens, timeout, streaming, structured_output

### Error handling
- Connection errors: gracefully skips analysis with warning
- Timeout errors: configurable timeout (default 120s)
- HTTP errors: returns error result without crashing

## License Compliance

All components use permissive licenses:
- yt-dlp: Unlicense
- OpenAI Whisper: MIT
- pyannote.audio: MIT
- torch: BSD

No proprietary or restricted models used.

## Known Limitations

- Whisper hallucinations: may generate text not spoken in audio (mitigated by beam_size=5 and context prompt)
- Speaker diarization: speaker labels are generic (speaker_0, speaker_1) — metadata context helps assign podcaster/guest labels
- YouTube channel monitoring: depends on yt-dlp extractor stability
- Long audio files: may need chunking for very long podcasts (>2 hours)
- CPU inference: ~2-3x slower than GPU, use medium model instead of turbo for better speed
- LLM analysis: requires local Ollama/LM Studio running; transcript truncation at 8000 chars for context limits

## Troubleshooting

### yt-dlp fails to download

- Verify ffmpeg is installed: `ffmpeg -version`
- Check YouTube URL is valid and public
- Update yt-dlp: `pip install -U yt-dlp`

### Whisper fails to load

- Verify CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU VRAM: `python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 1e9)"`
- If GPU < 6GB, switch model to "medium" in config.yaml

### pyannote.audio fails to load

- Verify HuggingFace token is valid and has read access
- Check pipeline name: `pyannote/speaker-diarization-community-1`
- Update pyannote.audio: `pip install -U pyannote.audio`

### SQLite storage errors

- Verify db directory exists
- Check file permissions
- Database auto-creates on first use

### LLM analysis unavailable

- Start Ollama: `ollama serve` (default port 11434)
- Start LM Studio: open app (default port 1234)
- Verify model is loaded: `ollama list` or LM Studio model selector
- Check config.yaml llm settings match your provider

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run unit tests only
python -m pytest tests/ -v -m "not integration"

# Run integration tests
python -m pytest tests/ -v -m "integration"
```

## Git History

Commits follow conventional format:
- `feat: ...` — new feature
- `test: ...` — new tests
- `fix: ...` — bug fix
- `docs: ...` — documentation updates
