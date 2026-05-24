"""SQLite storage layer for podcast transcripts and metadata — optimized for retrieval quality."""

import sqlite3
import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class PodcastStorage:
    """SQLite storage for podcast transcripts and metadata — optimized schema."""

    def __init__(self, db_path="data/podagent.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self._ensure_migrations()

    def _init_db(self):
        """Initialize database schema with optimized tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Podcasts table — with checksum and quality metrics
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
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                transcript_checksum TEXT,  -- SHA256 hash of transcript text for integrity
                transcription_confidence REAL,  -- Whisper confidence score
                diarization_quality REAL,  -- Diarization quality metric
                reprocessed INTEGER DEFAULT 0  -- Flag for reprocessing
            )
        """)

        # Transcript segments table — indexed for fast retrieval
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

        # Speakers table — indexed
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS speakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                podcast_id INTEGER REFERENCES podcasts(id),
                speaker_id TEXT,
                label TEXT,
                first_appearance REAL
            )
        """)

        # LLM analysis table — split structured fields into columns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                podcast_id INTEGER REFERENCES podcasts(id),
                analysis_mode TEXT,
                llm_model TEXT,
                provider TEXT,
                summary_text TEXT,
                topics TEXT,  -- JSON array of topics
                key_entities TEXT,  -- JSON array of key entities (people, places, orgs)
                key_points TEXT,  -- JSON array of key points
                sentiment TEXT,  -- Overall sentiment (positive/neutral/negative)
                insights_count INTEGER,  -- Number of insights extracted
                main_themes TEXT,  -- JSON array of main themes
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_quality REAL  -- Quality metric (0-1)
            )
        """)

        # TTS audio table — with source tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tts_audio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                podcast_id INTEGER REFERENCES podcasts(id),
                tts_provider TEXT,
                voice TEXT,
                audio_path TEXT,
                file_size INTEGER,
                source TEXT,  -- "LLM summary" or custom file path
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # FTS5 virtual table for full-text search across transcripts and analysis
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                podcast_id,
                transcript_text,
                analysis_text,
                topics,
                content=podcasts
            )
        """)

        # Database metadata table — version tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS db_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('schema_version', '2')")
        cursor.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('created_at', CURRENT_TIMESTAMP)")

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at: {self.db_path}")

    def _ensure_migrations(self):
        """Ensure schema migrations are applied."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check schema version
        cursor.execute("SELECT value FROM db_metadata WHERE key = 'schema_version'")
        result = cursor.fetchone()
        current_version = result[0] if result else "1"

        if current_version == "1":
            logger.info("Applying migration from schema v1 to v2")
            # Add new columns to podcasts table
            cursor.execute("ALTER TABLE podcasts ADD COLUMN transcript_checksum TEXT")
            cursor.execute("ALTER TABLE podcasts ADD COLUMN transcription_confidence REAL")
            cursor.execute("ALTER TABLE podcasts ADD COLUMN diarization_quality REAL")
            cursor.execute("ALTER TABLE podcasts ADD COLUMN reprocessed INTEGER DEFAULT 0")

            # Add new columns to llm_analysis table
            cursor.execute("ALTER TABLE llm_analysis ADD COLUMN topics TEXT")
            cursor.execute("ALTER TABLE llm_analysis ADD COLUMN key_entities TEXT")
            cursor.execute("ALTER TABLE llm_analysis ADD COLUMN key_points TEXT")
            cursor.execute("ALTER TABLE llm_analysis ADD COLUMN sentiment TEXT")
            cursor.execute("ALTER TABLE llm_analysis ADD COLUMN insights_count INTEGER")
            cursor.execute("ALTER TABLE llm_analysis ADD COLUMN main_themes TEXT")
            cursor.execute("ALTER TABLE llm_analysis ADD COLUMN analysis_quality REAL")

            # Add TTS source column
            cursor.execute("ALTER TABLE tts_audio ADD COLUMN source TEXT")

            # Create FTS5 search index
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                    podcast_id,
                    transcript_text,
                    analysis_text,
                    topics,
                    content=podcasts
                )
            """)

            # Create db_metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS db_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            cursor.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('schema_version', '2')")

            conn.commit()
            logger.info("Migration complete: schema v2")

        # Add indexes (only if they don't exist)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_podcast_id ON transcript_segments(podcast_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_podcast_id ON speakers(podcast_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_podcast_id ON llm_analysis(podcast_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_podcast_id ON tts_audio(podcast_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_channel_name ON podcasts(channel_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_mode ON llm_analysis(analysis_mode)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON llm_analysis(created_at)")

        conn.commit()
        conn.close()

    def save_podcast(self, podcast_data: dict) -> int:
        """Save podcast metadata to database. Returns the inserted or existing podcast ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if podcast already exists
        cursor.execute("SELECT id FROM podcasts WHERE video_id = ?", (podcast_data["video_id"],))
        existing = cursor.fetchone()
        if existing:
            podcast_id = existing[0]
            # Update existing podcast with new data (reprocessed flag)
            cursor.execute("""
                UPDATE podcasts SET
                    title = ?,
                    channel_id = ?,
                    channel_name = ?,
                    audio_path = ?,
                    transcript_path = ?,
                    language = ?,
                    duration = ?,
                    num_speakers = ?,
                    reprocessed = 1
                WHERE id = ?
            """, (
                podcast_data["title"],
                podcast_data["channel_id"],
                podcast_data["channel_name"],
                podcast_data["audio_path"],
                podcast_data["transcript_path"],
                podcast_data["language"],
                podcast_data["duration"],
                podcast_data["num_speakers"],
                podcast_id
            ))
            conn.commit()
            conn.close()
            logger.info(f"Podcast updated (reprocessed): video_id={podcast_data['video_id']}")
            return podcast_id

        # Calculate transcript checksum if transcript path is provided
        transcript_checksum = None
        if podcast_data.get("transcript_path") and os.path.isfile(podcast_data["transcript_path"]):
            with open(podcast_data["transcript_path"], "r") as f:
                transcript_text = f.read()
                transcript_checksum = hashlib.sha256(transcript_text.encode()).hexdigest()

        cursor.execute("""
            INSERT INTO podcasts
            (video_id, title, channel_id, channel_name, audio_path,
             transcript_path, language, duration, num_speakers,
             transcript_checksum, transcription_confidence, diarization_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            podcast_data["video_id"],
            podcast_data["title"],
            podcast_data["channel_id"],
            podcast_data["channel_name"],
            podcast_data["audio_path"],
            podcast_data["transcript_path"],
            podcast_data["language"],
            podcast_data["duration"],
            podcast_data["num_speakers"],
            transcript_checksum,
            podcast_data.get("transcription_confidence"),
            podcast_data.get("diarization_quality")
        ))
        conn.commit()
        podcast_id = cursor.lastrowid
        conn.close()
        logger.info(f"Podcast saved: video_id={podcast_data['video_id']}")
        return podcast_id

    def save_segments(self, podcast_id: int, segments: list):
        """Save transcript segments to database using batch insert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Clear old segments for this podcast
        cursor.execute("DELETE FROM transcript_segments WHERE podcast_id = ?", (podcast_id,))
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
        # Clear old speakers for this podcast
        cursor.execute("DELETE FROM speakers WHERE podcast_id = ?", (podcast_id,))
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
                "num_speakers": r[9], "processed_at": r[10],
                "transcript_checksum": r[11],
                "transcription_confidence": r[12],
                "diarization_quality": r[13],
                "reprocessed": r[14]
            }
            for r in rows
        ]

    def save_llm_analysis(self, podcast_id: int, analysis_result: dict):
        """Save LLM analysis results to database with structured fields."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Extract structured fields from analysis result
        topics = json.dumps(analysis_result.get("topics", [])) if analysis_result.get("topics") else None
        key_entities = json.dumps(analysis_result.get("key_entities", [])) if analysis_result.get("key_entities") else None
        key_points = json.dumps(analysis_result.get("key_points", [])) if analysis_result.get("key_points") else None
        sentiment = analysis_result.get("sentiment", "")
        insights_count = analysis_result.get("insights_count", 0)
        main_themes = json.dumps(analysis_result.get("main_themes", [])) if analysis_result.get("main_themes") else None
        analysis_quality = analysis_result.get("analysis_quality", 1.0)

        cursor.execute("""
            INSERT INTO llm_analysis
            (podcast_id, analysis_mode, llm_model, provider,
             summary_text, topics, key_entities, key_points,
             sentiment, insights_count, main_themes, processing_time, analysis_quality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            podcast_id,
            analysis_result.get("analysis_mode", ""),
            analysis_result.get("llm_model", ""),
            analysis_result.get("provider", ""),
            analysis_result.get("summary_text", ""),
            topics,
            key_entities,
            key_points,
            sentiment,
            insights_count,
            main_themes,
            analysis_result.get("processing_time_seconds", 0),
            analysis_quality
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
                "summary_text": r[5], "topics": r[6],
                "key_entities": r[7], "key_points": r[8],
                "sentiment": r[9], "insights_count": r[10],
                "main_themes": r[11], "processing_time": r[12],
                "analysis_quality": r[13], "created_at": r[14]
            }
            for r in rows
        ]

    def save_tts_audio(self, podcast_id: int, tts_result: dict):
        """Save TTS audio metadata to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tts_audio
            (podcast_id, tts_provider, voice, audio_path, file_size, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            podcast_id,
            tts_result.get("provider", ""),
            tts_result.get("voice", ""),
            tts_result.get("output_path", ""),
            tts_result.get("file_size", 0),
            tts_result.get("source", "LLM summary")
        ))
        conn.commit()
        conn.close()

    def get_tts_audio(self, podcast_id: Optional[int] = None) -> list[dict]:
        """Retrieve TTS audio records from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if podcast_id:
            cursor.execute(
                "SELECT * FROM tts_audio WHERE podcast_id = ? ORDER BY created_at DESC",
                (podcast_id,)
            )
        else:
            cursor.execute("SELECT * FROM tts_audio ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id": r[0], "podcast_id": r[1], "tts_provider": r[2],
                "voice": r[3], "audio_path": r[4],
                "file_size": r[5], "source": r[6], "created_at": r[7]
            }
            for r in rows
        ]

    def search_transcripts(self, query: str, limit: int = 10) -> list[dict]:
        """Search transcripts and analysis using FTS5 full-text search."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT podcast_id, transcript_text, analysis_text, topics
            FROM search_index
            WHERE search_index MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "podcast_id": r[0],
                "transcript_text": r[1],
                "analysis_text": r[2],
                "topics": r[3]
            }
            for r in rows
        ]

    def update_search_index(self, podcast_id: int):
        """Update FTS5 search index for a podcast."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get transcript text
        cursor.execute("SELECT text FROM transcript_segments WHERE podcast_id = ?", (podcast_id,))
        segments = cursor.fetchall()
        transcript_text = " ".join([seg[0] for seg in segments])

        # Get analysis text
        cursor.execute("SELECT summary_text FROM llm_analysis WHERE podcast_id = ?", (podcast_id,))
        analyses = cursor.fetchall()
        analysis_text = " ".join([analysis[0] for analysis in analyses])

        # Get topics
        cursor.execute("SELECT topics FROM llm_analysis WHERE podcast_id = ?", (podcast_id,))
        topics_row = cursor.fetchone()
        topics = topics_row[0] if topics_row else ""

        # Update search index
        cursor.execute("""
            INSERT INTO search_index (podcast_id, transcript_text, analysis_text, topics)
            VALUES (?, ?, ?, ?)
        """, (podcast_id, transcript_text, analysis_text, topics))
        conn.commit()
        conn.close()
        logger.info(f"Search index updated for podcast_id={podcast_id}")

    def verify_transcript_integrity(self, podcast_id: int) -> bool:
        """Verify transcript integrity by comparing checksum."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT transcript_checksum, transcript_path FROM podcasts WHERE id = ?", (podcast_id,))
        row = cursor.fetchone()
        conn.close()

        if not row or not row[1]:
            return False

        checksum = row[0]
        transcript_path = row[1]

        if not os.path.isfile(transcript_path):
            return False

        with open(transcript_path, "r") as f:
            transcript_text = f.read()
            current_checksum = hashlib.sha256(transcript_text.encode()).hexdigest()

        return checksum == current_checksum

    def get_db_stats(self) -> dict:
        """Return database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM podcasts")
        podcast_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM transcript_segments")
        segment_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM llm_analysis")
        analysis_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM tts_audio")
        tts_count = cursor.fetchone()[0]

        cursor.execute("SELECT value FROM db_metadata WHERE key = 'schema_version'")
        schema_version = cursor.fetchone()[0]

        conn.close()
        return {
            "podcast_count": podcast_count,
            "segment_count": segment_count,
            "analysis_count": analysis_count,
            "tts_count": tts_count,
            "schema_version": schema_version
        }
