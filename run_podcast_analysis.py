#!/usr/bin/env python3
"""Run full LLM analysis on a podcast transcript using chat completions with reasoning mode.

Key improvement: uses /v1/chat/completions instead of /v1/completions, which gives us clean
separation between the model's internal thinking (reasoning_content) and its final output (content).
No post-hoc regex cleanup needed — we just read content and discard reasoning_content entirely.

Usage:
    python3 run_podcast_analysis.py --folder <path_to_output_folder>
    # or if transcript is at expected location:
    python3 run_podcast_analysis.py --url https://www.youtube.com/watch?v=...
"""

import os, sys, logging, re, httpx, time, asyncio, argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("PodcastAnalysis")

import yaml

# Load config (LLM_URL from config.yaml, not hardcoded)
def _load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

_llm_cfg = None

def _get_llm_config():
    global _llm_cfg
    if _llm_cfg is None:
        cfg = _load_config()
        # Try settings.analysis first, fall back to settings.llm
        llm = cfg.get("settings", {}).get("analysis", {})
        if not llm or "provider" not in llm and "model" not in llm:
            llm = cfg.get("settings", {}).get("llm", {})
        provider = llm.get("provider", "lmstudio")
        url_key = "lmstudio_url" if provider == "lmstudio" else "base_url"
        _llm_cfg = {
            "url": llm.get(url_key, cfg.get("settings", {}).get("llm", {}).get(url_key, "http://192.168.1.50:1234")),
            "model": llm.get("model", cfg.get("settings", {}).get("llm", {}).get("model", "qwen/qwen3.6-35b-a3b-nvfp4")),
        }
    return _llm_cfg

# Initialize from config at module load time
LLM_URL = None  # set below
MODEL_NAME = None  # set below

def _init_llm_config():
    global LLM_URL, MODEL_NAME
    cfg = _get_llm_config()
    LLM_URL = cfg["url"]
    # For LM Studio, model is discovered at runtime from /v1/models.
    # MODEL_NAME is the config default — used only when no model is loaded in the UI.
    MODEL_NAME = cfg["model"]

_init_llm_config()


def find_transcript(folder_path):
    """Find the transcript markdown file in a PodAgent output folder."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    transcript_dir = os.path.join(folder_path, "transcript")
    if not os.path.isdir(transcript_dir):
        raise FileNotFoundError(f"No transcript directory in {folder_path}")
    
    md_files = [f for f in os.listdir(transcript_dir) if f.endswith("_transcript.md")]
    if not md_files:
        raise FileNotFoundError(f"No transcript file found in {transcript_dir}")
    
    return os.path.join(transcript_dir, sorted(md_files)[0])


def parse_md_transcript(file_path):
    """Parse the PodAgent markdown transcript into structured segments."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    with open(file_path, "r") as f:
        lines = f.readlines()

    video_title = "Unknown Video"
    speakers_found = []

    for line in lines:
        if line.startswith("# "):
            video_title = line.strip()[2:]
            continue
        # Format from file: - **Speaker Name** (first at 0.0m) or just - **Speaker Name**
        m_speaker = re.search(r"- \*\*(.+?)\*\*", line)
        if m_speaker and len(speakers_found) < 5:
            name = m_speaker.group(1).strip()
            # Skip metadata lines that happen to be bolded (Duration, Language, etc.)
            if not any(name.startswith(x) for x in ["Duration", "Language"]):
                speakers_found.append(name)

    segments = []
    for line in lines:
        ls = line.strip()
        
        # Skip metadata headers and speaker list entries
        skip_prefixes = ["# ", "**Duration:", "**Language:", "## Speakers", "- **"]
        if not ls or any(ls.startswith(p) for p in skip_prefixes):
            continue
        
        # Remove markdown bold markers (**), then split on first ": " (colon-space)
        clean_line = re.sub(r'^\*\*', '', ls)  # remove leading **
        parts = clean_line.split(': ', 1)     # split on ': ' 
        if len(parts) != 2: continue
        
        sp = parts[0].strip()                 # e.g. "Tom Bilyeu"  
        txt = parts[1].strip()

        ad_keywords = ["ethosis makes getting life insurance", "Speaker 2acity", 
                      "03-", "get a quote in seconds"]
        if any(kw.lower() in txt.lower().lower() for kw in ad_keywords): continue
        
        if len(txt) < 10 and sp not in ("Tom Bilyeu", "Guest 1", "Guest 2"): continue
                
        segments.append({
            "speaker_label": sp,
            "text": txt
        })

    return {
        "video_title": video_title,
        "speakers": speakers_found,
        "segments": segments
    }


def build_prompt(title, speakers, segments, mode):
    """Build the user message for a given analysis mode.
    
    Uses numbered sections which we know trigger Qwen's actual content generation.
    The system prompt handles suppressing meta-text, so these are purely structural guides.
    """
    speaker_info = "\n".join(f"  - {s}" for s in speakers) if speakers else "N/A"

    # Limit to first 200 segments to stay within context window
    max_segs = min(len(segments), 200)
    seg_text = "\n".join(f"[{s['speaker_label']}] {s['text']}" for s in segments[:max_segs])
    if len(segments) > max_segs:
        seg_text += f"\n\n[... {len(segments) - max_segs} more segments truncated ...]"

    prompts = {
        # Summary — direct prose, no outline thinking
        "summary": f"""Analyze this podcast transcript and provide a comprehensive summary.

Video Title: {title}
Speakers: {speaker_info}

Transcript (first 200 segments):
{seg_text}

Write your response using THIS exact numbered structure:

1. Executive Summary — 3-5 sentences describing what the video is about, 
   who the speakers are discussing, and their main conclusions
2. Key Topics Covered — bullet points of the major themes discussed
3. Main Arguments or Conclusions — what the speakers concluded
4. Important Timestamps — key sections with approximate timestamps""",

        # Insights — deep analysis, not surface-level
        "insights": f"""Extract deep insights from this podcast transcript.

Video Title: {title}
Speakers: {speaker_info}

Transcript (first 200 segments):
{seg_text}

Write detailed analysis covering these areas:

1. Key Takeaways — the most important points
2. Unconventional or Counterintuitive Points  
3. Actionable Insights for Viewers
4. Contextual Analysis of why these insights matter""",

        # Notes — structured study notes with specific content
        "notes": f"""Take detailed study notes from this podcast transcript.

Video Title: {title}
Speakers: {speaker_info}

Transcript (first 200 segments):
{seg_text}

Write structured notes using numbered sections:

1. Topic-by-topic breakdown of the discussion — list each major topic as a 
   numbered item with bullet points underneath
2. Key definitions and concepts explained — define terms like yield curve, 
   Eurodollar system, K-shaped economy, etc.
3. Important data points or statistics mentioned — specific numbers or facts
4. Speaker attributions for key claims — who said what""",
    }
    
    return prompts.get(mode, prompts["summary"])


def build_system_prompt(mode):
    """Build the system prompt that controls how the model behaves.
    
    This is where we suppress internal planning/meta-text at the source.
    The model's reasoning goes into reasoning_content (which we discard),
    and content should be pure analysis output.
    """
    base = (
        "You are an expert podcast analyst. Your job is to produce direct, substantive "
        "prose analysis — not outlines, not planning notes, not bullet-point thinking."
    )

    system_prompts = {
        "summary": f"""{base}. Write a comprehensive summary in flowing prose with clear numbered sections as requested. Do not include your internal reasoning or planning process in the output — only write the final analysis.""",
        
        "insights": f"""{base}. Extract and explain deep insights using detailed paragraphs under each section heading. Support claims with specific quotes from the transcript when possible. Write substantive prose, not outline bullets.""",
        
        "notes": f"""{base}. Take detailed study notes organized by topic. Use numbered sections as requested but fill each section with concrete details — definitions, data points, speaker attributions, and direct quotes. Do NOT produce a meta-outline of topics; write the actual substantive notes about what was discussed.""",
    }
    
    return system_prompts.get(mode, system_prompts["summary"])


async def _resolve_active_model(url):
    """Query LM Studio /v1/models and return the currently loaded model ID.
    
    Falls back to None if discovery fails (caller should use config default).
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{url}/v1/models")
            if resp.status_code == 200 and resp.json().get("data"):
                return resp.json()["data"][0]["id"]
    except Exception as e:
        logger.warning(f"Failed to query /v1/models for active model: {e}")
    return None

async def call_llm(system_prompt, user_prompt, temperature=0.7):
    """Call the LLM via chat completions API with reasoning mode.
    
    Returns (content, raw_response) where content is the clean output and
    raw_response includes metadata for debugging.
    """
    # Discover the active model from LM Studio /v1/models; fall back to config default
    active_model = await _resolve_active_model(LLM_URL) or MODEL_NAME
    if not active_model:
        logger.error("No model available — could not discover active model and no config default")
        return "", "No model loaded in LM Studio and no configured fallback"

    payload = {
        "model": active_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{LLM_URL}/v1/chat/completions", json=payload)
        
        if resp.status_code != 200:
            return "", f"HTTP {resp.status_code}: {resp.text[:500]}"
        
        data = resp.json()
        
        # Extract content from chat completions response
        choices = data.get("choices", [])
        if not choices:
            return "", "No choices in response"
        
        msg = choices[0].get("message", {})
        content = msg.get("content", "").strip()
        reasoning = msg.get("reasoning_content", "")
        finish_reason = choices[0].get("finish_reason", "")
        
        # Log reasoning stats for debugging
        if reasoning:
            logger.debug(f"  Reasoning tokens discarded: {len(reasoning)} chars")
        
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", "?")
        completion_tokens = usage.get("completion_tokens", "?")
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        
        return content, {
            "content_length": len(content),
            "reasoning_length": len(reasoning),
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        }


async def run_analysis(folder_path):
    logger.info("=== PodAgent: Full Analysis Pipeline ===")
    logger.info(f"Folder: {folder_path}")
    logger.info(f"Model:  {MODEL_NAME}")
    logger.info(f"API:    chat completions with reasoning mode (content only, reasoning discarded)")

    # Find and parse transcript
    try:
        transcript_file = find_transcript(folder_path)
        transcript = parse_md_transcript(transcript_file)
        logger.info(f"Loaded '{transcript['video_title']}' — {len(transcript['segments'])} segments, speakers={transcript['speakers']}")
    except Exception as e:
        logger.error(f"Failed to load transcript: {e}")
        return

    # Test LLM connectivity first
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{LLM_URL}/v1/models")
            models_data = resp.json().get("data", [])
            active_model = models_data[0]["id"] if models_data else None
            if active_model:
                logger.info(f"Available models: {len(models_data)} (using loaded model '{active_model}')")
            else:
                logger.warning("No model loaded in LM Studio — will use configured default from config.yaml")
    except Exception as e:
        logger.error(f"LLM connectivity check failed: {e}")
        return

    # Run all 3 modes sequentially with retry on empty output
    for mode in ["summary", "insights", "notes"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING: {mode.upper()}")
        logger.info(f"{'='*60}")
        
        start = time.time()
        
        system_prompt = build_system_prompt(mode)
        user_prompt = build_prompt(
            transcript["video_title"], 
            transcript.get("speakers", []), 
            transcript["segments"], 
            mode
        )

        # First attempt with standard temperature
        cleaned, stats = await call_llm(system_prompt, user_prompt, temperature=0.7)

        elapsed = time.time() - start
        
        # If empty or too short, retry with lower temperature and modified system prompt
        if not cleaned or len(cleaned) < 50:
            logger.info(f"Response too short ({len(cleaned)} chars), retrying with lower temperature...")
            
            # More explicit system prompt for retry
            retry_system = f"{system_prompt}. IMPORTANT: Write your response DIRECTLY. Do not include any planning, thinking process, or outline — only the final analysis output."
            
            cleaned, stats = await call_llm(retry_system, user_prompt, temperature=0.3)

        logger.info(f"{mode.upper()} complete: {len(cleaned)} chars, {elapsed:.1f}s")
        if stats:
            logger.info(f"  Prompt tokens: {stats['prompt_tokens']}, Completion: {stats['completion_tokens']} (Reasoning discarded: {stats.get('reasoning_tokens', 'N/A')})")

        # Save to file
        output_path = os.path.join(folder_path, f"analysis_{mode}.md")
        with open(output_path, "w") as f:
            if cleaned:
                f.write(f"# Analysis Mode: {mode}\n\n")
                f.write(f"**Video:** {transcript['video_title']}\n")
                f.write(f"\n---\n\n{cleaned}\n")
            else:
                f.write(f"# Analysis Mode: {mode} — FAILED\n\n")
                if stats:
                    f.write(f"Content was empty. Stats: {stats}\n")

        logger.info(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PodAgent podcast analysis with LLM")
    parser.add_argument("--folder", help="Path to PodAgent output folder containing transcript/")
    args = parser.parse_args()
    
    if not args.folder:
        print("Usage: python3 run_podcast_analysis.py --folder <path_to_output_folder>")
        print("\nExample:")
        print("  python3 run_podcast_analysis.py --folder data/output_20260716_My_Podcast_Title")
        sys.exit(1)
    
    asyncio.run(run_analysis(args.folder))


if __name__ == "__main__":
    main()
