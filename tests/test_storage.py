import pytest
import tempfile
import os
import sqlite3
from src.storage import PodcastStorage


def test_storage_initialization():
    storage = PodcastStorage()
    assert storage.db_path is not None
    assert os.path.isfile(storage.db_path)


def test_storage_custom_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = PodcastStorage(db_path=db_path)
        assert storage.db_path == db_path
        assert os.path.isfile(db_path)


def test_storage_schema_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = PodcastStorage(db_path=db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check podcasts table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='podcasts'")
        assert cursor.fetchone() is not None

        # Check transcript_segments table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transcript_segments'")
        assert cursor.fetchone() is not None

        # Check speakers table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='speakers'")
        assert cursor.fetchone() is not None

        conn.close()


def test_storage_save_podcast():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = PodcastStorage(db_path=db_path)

        podcast_data = {
            "video_id": "test_video_1",
            "title": "Test Podcast 1",
            "channel_id": "UC_test",
            "channel_name": "Test Channel",
            "audio_path": "/tmp/test_audio.mp3",
            "transcript_path": "/tmp/test_transcript.json",
            "language": "en",
            "duration": 3600.0,
            "num_speakers": 2,
        }

        storage.save_podcast(podcast_data)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM podcasts WHERE video_id='test_video_1'")
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "test_video_1"  # video_id
        assert row[2] == "Test Podcast 1"  # title
        assert row[7] == "en"  # language
        assert row[8] == 3600.0  # duration
        assert row[9] == 2  # num_speakers
        conn.close()


def test_storage_save_segments():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = PodcastStorage(db_path=db_path)

        podcast_data = {
            "video_id": "test_video_1",
            "title": "Test Podcast 1",
            "channel_id": "UC_test",
            "channel_name": "Test Channel",
            "audio_path": "/tmp/test_audio.mp3",
            "transcript_path": "/tmp/test_transcript.json",
            "language": "en",
            "duration": 3600.0,
            "num_speakers": 2,
        }
        storage.save_podcast(podcast_data)

        segments = [
            {"start": 0.0, "end": 5.0, "speaker_label": "Podcaster", "text": "Welcome"},
            {"start": 5.0, "end": 10.0, "speaker_label": "Guest 1", "text": "Hello"},
        ]

        storage.save_segments(1, segments)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM transcript_segments WHERE podcast_id=1")
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0][2] == 0.0  # start_time
        assert rows[0][3] == 5.0  # end_time
        assert rows[0][4] == "Podcaster"  # speaker_label
        assert rows[0][5] == "Welcome"  # text
        conn.close()


def test_storage_save_speakers():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = PodcastStorage(db_path=db_path)

        podcast_data = {
            "video_id": "test_video_1",
            "title": "Test Podcast 1",
            "channel_id": "UC_test",
            "channel_name": "Test Channel",
            "audio_path": "/tmp/test_audio.mp3",
            "transcript_path": "/tmp/test_transcript.json",
            "language": "en",
            "duration": 3600.0,
            "num_speakers": 2,
        }
        storage.save_podcast(podcast_data)

        speakers = [
            {"speaker_id": "speaker_0", "label": "Podcaster", "first_appearance": 0.0},
            {"speaker_id": "speaker_1", "label": "Guest 1", "first_appearance": 5.0},
        ]

        storage.save_speakers(1, speakers)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM speakers WHERE podcast_id=1")
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0][2] == "speaker_0"  # speaker_id
        assert rows[0][3] == "Podcaster"  # label
        assert rows[0][4] == 0.0  # first_appearance
        conn.close()


def test_storage_get_all_podcasts():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = PodcastStorage(db_path=db_path)

        # Save multiple podcasts
        podcast_data_1 = {
            "video_id": "test_video_1",
            "title": "Test Podcast 1",
            "channel_id": "UC_test",
            "channel_name": "Test Channel",
            "audio_path": "/tmp/test_audio_1.mp3",
            "transcript_path": "/tmp/test_transcript_1.json",
            "language": "en",
            "duration": 3600.0,
            "num_speakers": 2,
        }
        podcast_data_2 = {
            "video_id": "test_video_2",
            "title": "Test Podcast 2",
            "channel_id": "UC_test2",
            "channel_name": "Test Channel 2",
            "audio_path": "/tmp/test_audio_2.mp3",
            "transcript_path": "/tmp/test_transcript_2.json",
            "language": "en",
            "duration": 1800.0,
            "num_speakers": 1,
        }

        storage.save_podcast(podcast_data_1)
        storage.save_podcast(podcast_data_2)

        all_podcasts = storage.get_all_podcasts()
        assert len(all_podcasts) == 2
        assert all_podcasts[0]["title"] == "Test Podcast 1"
        assert all_podcasts[1]["title"] == "Test Podcast 2"


def test_storage_duplicate_podcast_ignore():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = PodcastStorage(db_path=db_path)

        podcast_data = {
            "video_id": "test_video_1",
            "title": "Test Podcast 1",
            "channel_id": "UC_test",
            "channel_name": "Test Channel",
            "audio_path": "/tmp/test_audio.mp3",
            "transcript_path": "/tmp/test_transcript.json",
            "language": "en",
            "duration": 3600.0,
            "num_speakers": 2,
        }

        # Save same podcast twice
        storage.save_podcast(podcast_data)
        storage.save_podcast(podcast_data)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM podcasts WHERE video_id='test_video_1'")
        count = cursor.fetchone()[0]
        assert count == 1  # INSERT OR IGNORE prevents duplicates
        conn.close()
