"""SQLite storage layer for podcast transcripts and metadata."""

import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class PodcastStorage:
    """SQLite storage for podcast transcripts and metadata."""

    def __init__(self, db_path="data/podagent.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS podcasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT UNIQUE,
                title TEXT,
                channel_id TEXT,
                channel_name TEXT,
                audio_path TEXT,
                transcript_path TEXT,
                language TEXT,
                duration REAL,
                num_speakers INTEGER,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcript_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                podcast_id INTEGER REFERENCES podcasts(id),
                start_time REAL,
                end_time REAL,
                speaker_label TEXT,
                text TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS speakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                podcast_id INTEGER REFERENCES podcasts(id),
                speaker_id TEXT,
                label TEXT,
                first_appearance REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                podcast_id INTEGER REFERENCES podcasts(id),
                analysis_mode TEXT,
                llm_model TEXT,
                provider TEXT,
                summary_text TEXT,
                structured_output TEXT,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at: {self.db_path}")

    def save_podcast(self, podcast_data: dict) -> int:
        """Save podcast metadata to database. Returns the inserted or existing podcast ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if podcast already exists (INSERT OR IGNORE won't update)
        cursor.execute("SELECT id FROM podcasts WHERE video_id = ?", (podcast_data["video_id"],))
        existing = cursor.fetchone()
        if existing:
            podcast_id = existing[0]
            conn.close()
            return podcast_id

        cursor.execute("""
            INSERT INTO podcasts
            (video_id, title, channel_id, channel_name, audio_path,
             transcript_path, language, duration, num_speakers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            podcast_data["video_id"],
            podcast_data["title"],
            podcast_data["channel_id"],
            podcast_data["channel_name"],
            podcast_data["audio_path"],
            podcast_data["transcript_path"],
            podcast_data["language"],
            podcast_data["duration"],
            podcast_data["num_speakers"]
        ))
        conn.commit()
        podcast_id = cursor.lastrowid
        conn.close()
        return podcast_id

    def save_segments(self, podcast_id: int, segments: list):
        """Save transcript segments to database using batch insert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        rows = []
        for seg in segments:
            start = seg["start"] if isinstance(seg, dict) else seg.start
            end = seg["end"] if isinstance(seg, dict) else seg.end
            speaker_label = seg["speaker_label"] if isinstance(seg, dict) else seg.speaker_label
            text = seg["text"] if isinstance(seg, dict) else seg.text
            rows.append((podcast_id, start, end, speaker_label, text))
        cursor.executemany("""
            INSERT INTO transcript_segments
            (podcast_id, start_time, end_time, speaker_label, text)
            VALUES (?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(rows)} segments to DB for podcast_id={podcast_id}")

    def save_speakers(self, podcast_id: int, speakers: list):
        """Save speaker profiles to database using batch insert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        rows = []
        for sp in speakers:
            speaker_id = sp["speaker_id"] if isinstance(sp, dict) else sp.speaker_id
            label = sp["label"] if isinstance(sp, dict) else sp.label
            first_appearance = sp["first_appearance"] if isinstance(sp, dict) else sp.first_appearance
            rows.append((podcast_id, speaker_id, label, first_appearance))
        cursor.executemany("""
            INSERT INTO speakers
            (podcast_id, speaker_id, label, first_appearance)
            VALUES (?, ?, ?, ?)
        """, rows)
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(rows)} speakers to DB for podcast_id={podcast_id}")

    def get_all_podcasts(self) -> list[dict]:
        """Retrieve all podcasts from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM podcasts ORDER BY processed_at DESC")
        rows = cursor.fetchall()
        conn.close()
        # Convert rows to dicts
        return [
            {
                "id": r[0], "video_id": r[1], "title": r[2],
                "channel_id": r[3], "channel_name": r[4],
                "audio_path": r[5], "transcript_path": r[6],
                "language": r[7], "duration": r[8],
                "num_speakers": r[9], "processed_at": r[10]
            }
            for r in rows
        ]

    def save_llm_analysis(self, podcast_id: int, analysis_result: dict):
        """Save LLM analysis results to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO llm_analysis
            (podcast_id, analysis_mode, llm_model, provider,
             summary_text, structured_output, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            podcast_id,
            analysis_result.get("analysis_mode", ""),
            analysis_result.get("llm_model", ""),
            analysis_result.get("provider", ""),
            analysis_result.get("summary_text", ""),
            json.dumps(analysis_result.get("structured_output", {})) if analysis_result.get("structured_output") else None,
            analysis_result.get("processing_time_seconds", 0),
        ))
        conn.commit()
        conn.close()

    def get_llm_analyses(self, podcast_id: Optional[int] = None) -> list[dict]:
        """Retrieve LLM analysis results from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if podcast_id:
            cursor.execute(
                "SELECT * FROM llm_analysis WHERE podcast_id = ? ORDER BY created_at DESC",
                (podcast_id,)
            )
        else:
            cursor.execute("SELECT * FROM llm_analysis ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": r[0], "podcast_id": r[1], "analysis_mode": r[2],
                "llm_model": r[3], "provider": r[4],
                "summary_text": r[5], "structured_output": r[6],
                "processing_time": r[7], "created_at": r[8]
            }
            for r in rows
        ]
