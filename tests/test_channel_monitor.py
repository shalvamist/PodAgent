import pytest
import tempfile
import os
import json
import yaml
from src.channel_monitor import ChannelMonitor


def test_channel_monitor_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        channels_file = os.path.join(tmpdir, "channels.yaml")
        with open(channels_file, "w") as f:
            yaml.dump({"channels": [{"id": "UC_test", "name": "Test Channel"}]}, f)
        monitor = ChannelMonitor(
            channels_file=channels_file,
            storage_dir=tmpdir
        )
        assert monitor is not None
        assert len(monitor.channels) == 1
        assert monitor.channels[0]["id"] == "UC_test"


def test_channel_monitor_load_channels():
    with tempfile.TemporaryDirectory() as tmpdir:
        channels_file = os.path.join(tmpdir, "channels.yaml")
        with open(channels_file, "w") as f:
            yaml.dump({
                "channels": [
                    {"id": "UC_1", "name": "Podcast 1"},
                    {"id": "UC_2", "name": "Podcast 2"},
                ]
            }, f)
        monitor = ChannelMonitor(channels_file=channels_file, storage_dir=tmpdir)
        assert len(monitor.channels) == 2
        assert monitor.channels[0]["name"] == "Podcast 1"


def test_channel_monitor_processed_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        channels_file = os.path.join(tmpdir, "channels.yaml")
        with open(channels_file, "w") as f:
            yaml.dump({"channels": [{"id": "UC_test", "name": "Test"}]}, f)
        monitor = ChannelMonitor(channels_file=channels_file, storage_dir=tmpdir)
        assert monitor._processed_videos == {}


def test_channel_monitor_save_processed_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        channels_file = os.path.join(tmpdir, "channels.yaml")
        with open(channels_file, "w") as f:
            yaml.dump({"channels": [{"id": "UC_test", "name": "Test"}]}, f)
        monitor = ChannelMonitor(channels_file=channels_file, storage_dir=tmpdir)
        monitor._processed_videos["video_1"] = {"processed_at": "2026-01-01"}
        monitor._save_processed_log()
        log_path = os.path.join(tmpdir, "processed_log.json")
        assert os.path.exists(log_path)
        with open(log_path, "r") as f:
            data = json.load(f)
        assert data["video_1"]["processed_at"] == "2026-01-01"


def test_channel_monitor_monitor_all_channels_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        channels_file = os.path.join(tmpdir, "channels.yaml")
        with open(channels_file, "w") as f:
            yaml.dump({"channels": []}, f)
        monitor = ChannelMonitor(channels_file=channels_file, storage_dir=tmpdir)
        new_videos = monitor.monitor_all_channels()
        assert new_videos == []
