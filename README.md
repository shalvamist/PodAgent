# PodAgent

A system for downloading, transcribing, and analyzing YouTube podcast audio.

## Features

- Auto-monitor YouTube channels for new podcast uploads
- Download audio using yt-dlp
- Transcribe audio using Whisper turbo (open-source, MIT license)
- Speaker diarization using pyannote.audio (open-source, MIT license)
- Generate structured transcripts with podcaster/guest identification

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Install ffmpeg: `sudo apt install ffmpeg`
3. Create HuggingFace token: https://hf.co/settings/tokens
4. Add channels to `data/channels.yaml`
5. Set your HF token in `config.yaml`
6. Run: `python run.py`

## Usage

```bash
# Process a single video
python run.py --url https://www.youtube.com/watch?v=VIDEO_ID

# Monitor all configured channels
python run.py --monitor
```

## Architecture

Three-stage pipeline:
1. **Download** — yt-dlp extracts audio + metadata from YouTube
2. **Transcribe** — Whisper turbo converts audio to text with context-enhanced prompting
3. **Diarize** — pyannote.audio identifies speakers with timestamped segments

## GPU Strategy

- Primary: NVIDIA GPU via CUDA (torch.cuda.is_available())
- Fallback: CPU inference with fp32 mode (~2-3x slower)
- GPU < 6GB VRAM: auto-switches to medium model instead of turbo

## License Compliance

All components use permissive licenses (MIT/Unlicense). No proprietary or restricted models used.
