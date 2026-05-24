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
|- Optional local LLM analysis (Ollama/LM Studio): summary, insights, notes, blog
|- Text-to-speech audio generation from LLM summaries or custom text files

## Architecture

```
YouTube Channel --> ChannelMonitor --> yt-dlp (audio + metadata)
                                      --> WhisperTranscriber (text + segments)
                                      --> SpeakerDiarizer (speaker labels)
                                      --> TranscriptBuilder (structured output)
                                      --> PodcastStorage (SQLite database)
                                      --> LLMAnalyzer (optional: summary/insights/notes/blog)
                                      --> TTSGenerator (optional: audio from summary or custom file)
```

Three-stage pipeline with metadata context:
1. **Download** — yt-dlp extracts audio + metadata from YouTube
2. **Transcribe** — Whisper turbo converts audio to text with context-enhanced prompting (guest names from description, tags, channel topic)
3. **Diarize** — pyannote.audio identifies speakers with timestamped segments
4. **Build** — combines transcription + diarization into structured transcript with speaker labels
5. **Store** — saves to SQLite database
6. **Analyze** — (optional) feeds transcript to local LLM for summary/insights/notes/blog
7. **TTS** — (optional) generates audio summary from LLM output or custom text file

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

### Usage

#### Process a single video

```bash
python run.py --url https://www.youtube.com/watch?v=VIDEO_ID
```

#### Monitor all configured channels

```bash
python run.py --monitor
```

#### Custom config file

```bash
python run.py --config /path/to/config.yaml
```

#### Help

```bash
python run.py --help
```

#### Run LLM analysis on transcripts

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

#### Generate TTS audio summary

```bash
# Use LLM summary text (requires --analyze)
python run.py --url https://www.youtube.com/watch?v=VIDEO_ID --analyze --tts

# Use a custom file as TTS source
python run.py --url https://www.youtube.com/watch?v=VIDEO_ID --analyze --tts /path/to/file.md
```

`--tts` accepts two modes:
- **No argument** (`--tts`) — reads the LLM summary text from the first analysis result
- **File path** (`--tts <file_path>`) — reads the specified file and generates TTS from its content

TTS audio saved to `data/<output_dir>/tts/` as `{video_id}_tts.mp3`.

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
│   ├── tts_generator.py     # Text-to-speech audio generation
│   └── utils.py             # Shared utilities
├── data/
│   ├── audio/               # Downloaded audio files
│   ├── transcripts/         # Generated transcript JSON files
│   ├── llm_analysis/        # LLM analysis results
│   ├── tts/                 # Generated TTS audio files
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
### podcasts table
- id (autoincrement), video_id (UNIQUE)
- title, channel_id, channel_name
- audio_path, transcript_path
- language, duration, num_speakers
- processed_at (timestamp)
- transcript_checksum (SHA256 hash for integrity verification)
- transcription_confidence (Whisper confidence score)
- diarization_quality (0-1 metric)
- reprocessed (flag for reprocessing)

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
- summary_text
- topics (JSON array of topics)
- key_entities (JSON array of entities: people, places, orgs)
- key_points (JSON array of key points)
- sentiment (positive/neutral/negative)
- insights_count (number of insights)
- main_themes (JSON array of themes)
- processing_time (seconds)
- analysis_quality (0-1 metric)
- created_at (timestamp)

### tts_audio table
- podcast_id (FK to podcasts)
- audio_path
- tts_provider, voice
- source (LLM summary or custom file path)
- file_size (bytes)
- created_at (timestamp)

### search_index (FTS5 virtual table)
- podcast_id, transcript_text, analysis_text, topics
- Enables full-text search across all transcripts and analysis

### db_metadata table
- schema_version (version tracking)
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

## TTS Generation

Text-to-speech audio from transcript summaries or custom text files.

### Supported providers
- **edge-tts** — free Microsoft Edge TTS (default)
- **elevenlabs** — ElevenLabs API (requires API key)

### Configuration
```yaml
settings:
  tts:
    provider: edge-tts
    voice: en-US-AvaMultilingualNeural
    rate: "+0%"
    pitch: "+0Hz"
    output_format: mp3
    elevenlabs_api_key: ""
    elevenlabs_model: eleven_multilingual_v2
```

### TTS modes
- **LLM summary** (`--tts` no argument) — uses the first LLM analysis result's summary text
- **Custom file** (`--tts <file_path>`) — reads the specified file and generates TTS from its content

## Database Features

### Full-Text Search (FTS5)
- `search_index` virtual table indexes all transcript text + LLM analysis text + topics
- Enables keyword search across hundreds of podcasts without reading each file
- Example: `search podcasts where transcript contains "Bob Lazar UFO"`

### Quality Metrics
- **Transcription confidence** — average Whisper segment confidence score (0-1)
- **Diarization quality** — heuristic based on segment count and speaker distribution (0-1)
- **Analysis quality** — LLM analysis quality metric (0-1)

### Data Integrity
- **Transcript checksum** — SHA256 hash of transcript text stored in DB
- Enables verification of transcript integrity against disk files
- Detects corruption or mismatch after disk failures

### Structured JSON Fields
- LLM analysis stored as separate columns: topics, key_entities, key_points, sentiment, main_themes
- Enables filtering/aggregation across podcasts: "find all podcasts with sentiment='negative'"

### Migration System
- Schema version tracking in `db_metadata` table
- Handles schema upgrades without data loss
- Automatic migration from v1 to v2 on startup

### Indexes
- FK indexes on transcript_segments, speakers, llm_analysis, tts_audio
- channel_name index on podcasts
- analysis_mode index on llm_analysis
- created_at index on llm_analysis

### Output
- Saved to `data/<output_dir>/tts/{video_id}_tts.mp3`
- Stored in `podagent.db` (tts_audio table) with provider, voice, source, and file size metadata

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
- TTS generation: requires `--analyze` for LLM summary mode; custom file mode works independently

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

### TTS generation fails

- Verify tts config section exists in config.yaml
- Check voice name is valid for the provider
- For ElevenLabs: verify API key is set and account has credits
- edge-tts requires internet connection (uses Microsoft Edge TTS API)

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
