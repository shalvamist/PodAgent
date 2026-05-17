"""SQLite storage layer for podcast transcripts and metadata."""

import sqlite3
import os
import logging
from datetime import datetime

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

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at: {self.db_path}")

    def save_podcast(self, podcast_data: dict):
        """Save podcast metadata to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO podcasts
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
        conn.close()

    def save_segments(self, podcast_id: int, segments: list):
        """Save transcript segments to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for seg in segments:
            cursor.execute("""
                INSERT INTO transcript_segments
                (podcast_id, start_time, end_time, speaker_label, text)
                VALUES (?, ?, ?, ?, ?)
            """, (podcast_id, seg["start"], seg["end"], seg["speaker_label"], seg["text"]))
        conn.commit()
        conn.close()

    def save_speakers(self, podcast_id: int, speakers: list):
        """Save speaker profiles to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for sp in speakers:
            cursor.execute("""
                INSERT INTO speakers
                (podcast_id, speaker_id, label, first_appearance)
                VALUES (?, ?, ?, ?)
            """, (podcast_id, sp["speaker_id"], sp["label"], sp["first_appearance"]))
        conn.commit()
        conn.close()

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
