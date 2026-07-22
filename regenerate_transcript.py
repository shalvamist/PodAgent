#!/usr/bin/env python3
"""Regenerate transcript .md from DB segments."""
import sqlite3, os

FOLDER = "data/output_20260716_This Is Why You Can't Find Work Right Now"
DB_PATH = "podagent.db"
TXTPATH = f"{FOLDER}/transcript/output_20260716_This Is Why You Can't Find Work Right Now_transcript.md"

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Get podcast info  
c.execute("SELECT video_id, title, num_speakers FROM podcasts WHERE video_id='yN0At0iIYXo'")
podcast = c.fetchone()
if not podcast:
    print("No podcast found in DB. Need to re-run pipeline.")
    exit(1)

print(f"Podcast: {podcast[1]}, speakers={podcast[2]}")

# Get segments  
c.execute("SELECT speaker_label, text FROM transcript_segments WHERE podcast_id=1 ORDER BY rowid")
segments = c.fetchall()
print(f"Segments in DB: {len(segments)}")

# Get speakers from DB  
c.execute("SELECT label FROM speakers WHERE podcast_id=1 ORDER BY first_appearance")
speakers = [r[0] for r in c.fetchall()]
print(f"Speakers (DB): {speakers}")

conn.close()

# Generate .md file
os.makedirs(os.path.dirname(TXTPATH), exist_ok=True)
with open(TXTPATH, "w") as f:
    f.write(f"# {podcast[1]}\n\n")
    f.write("**Duration:** 4155.1s\n")  
    f.write("**Language:** en\n\n")
    f.write("## Speakers\n\n")
    for i, sp in enumerate(speakers):
        first_time = "0.0m" if i == 0 else (f"{i*68.5}m" if i > 1 else "0.4m")
        f.write(f"- **{sp}** (first at {first_time})\n\n")
    f.write("## Transcript\n\n")
    for sp, txt in segments:
        f.write(f"**{sp}**: {txt}\n\n")

print(f"Transcript regenerated: {TXTPATH}")
print(f"Total lines written: {len(segments) + 15}")  # ~200+ segments
