"""Transcript builder module — combines transcription + diarization into structured output."""

import json
import os
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    """Profile of a speaker identified in the transcript."""
    speaker_id: str  # e.g., "speaker_0"
    label: Optional[str]  # e.g., "Podcaster", "Guest 1", "John Doe"
    first_appearance: float  # timestamp in seconds


@dataclass
class StructuredTranscript:
    """A complete structured transcript with speaker labels and metadata."""
    video_title: str
    audio_path: str
    language: str
    duration: float
    speakers: list[SpeakerProfile]
    segments: list  # [{start, end, speaker_label, text}]
    raw_text: str
    output_path: str


class TranscriptBuilder:
    """Combine transcription and diarization into structured transcript with metadata context."""

    def __init__(self, base_data_dir: str = "data"):
        self.base_data_dir = base_data_dir
        os.makedirs(base_data_dir, exist_ok=True)

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename."""
        invalid_chars = '<>:"/\\|?*'''
        for c in invalid_chars:
            name = name.replace(c, "_")
        name = name.strip()
        return name[:100]

    def _get_video_folder(self, title: str) -> str:
        """Get or create the per-video data folder using structured naming."""
        from src.folder_manager import get_video_folder
        return get_video_folder(self.base_data_dir, title)

    def extract_guest_names_from_metadata(self, metadata) -> dict[str, str]:
        """Extract guest names from YouTube description and assign speaker labels."""
        guest_to_label = {}
        if hasattr(metadata, "description") and metadata.description:
            desc = metadata.description
            # Common patterns in YouTube descriptions
            patterns = [
                # "with John Doe", "featuring Jane Smith", "guest Bob"
                r'(?:with|featuring|guest|guests?|special guest|joined by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                # "John Doe discusses", "Jane Smith appears"
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:is\s+|joined\s+|appears\s+|discusses|talks\s+about|explains)',
            ]
            guest_names = []
            for pattern in patterns:
                matches = re.findall(pattern, desc, re.IGNORECASE)
                guest_names.extend([name.strip() for name in matches])

            # Deduplicate and assign labels
            seen = set()
            for name in guest_names:
                if name not in seen:
                    seen.add(name)
                    guest_num = len(guest_to_label) + 1
                    guest_to_label[name] = f"Guest {guest_num}"

        logger.info(f"Extracted {len(guest_to_label)} guest names from metadata")
        return guest_to_label

    def _assign_speaker_labels(
        self,
        diarization_segments: list,
        metadata,
        podcaster_speaker: Optional[str],
        guest_names: dict[str, str],
    ) -> dict[str, str]:
        """Assign speaker labels based on diarization order and metadata context.

        Strategy:
        1. First speaker (earliest appearance) = Podcaster (or uploader name)
        2. Remaining speakers = Guest 1, Guest 2, etc.
        3. If guest names extracted from metadata, we store them for reference
           but cannot reliably map them to speaker IDs from diarization alone.
        """
        speaker_labels = {}
        extracted_guests = {}

        # Sort segments by start time to determine speaker order
        sorted_segments = sorted(diarization_segments, key=lambda s: s["start"])

        # Assign first speaker as Podcaster
        first_segment = sorted_segments[0] if sorted_segments else None
        if first_segment:
            first_speaker = first_segment["speaker"]
            if metadata and hasattr(metadata, "uploader") and metadata.uploader:
                speaker_labels[first_speaker] = metadata.uploader
            else:
                speaker_labels[first_speaker] = "Podcaster"

        # Assign remaining speakers as Guests
        guest_num = 1
        for seg in sorted_segments[1:]:
            speaker_id = seg["speaker"]
            if speaker_id not in speaker_labels:
                speaker_labels[speaker_id] = f"Guest {guest_num}"
                guest_num += 1

        logger.info(f"Assigned speaker labels: {speaker_labels}")
        logger.info(f"Extracted guest names from metadata: {guest_names}")
        return speaker_labels

    def _find_speaker_for_segment(
        self,
        trans_seg_start: float,
        diarization_segments: list,
    ) -> Optional[str]:
        """Find the speaker for a transcription segment's time range from diarization."""
        for diag_seg in diarization_segments:
            if diag_seg["start"] <= trans_seg_start < diag_seg["end"]:
                return diag_seg["speaker"]
        return None

    def build(
        self,
        transcription_result,
        diarization_result,
        video_title: str,
        video_id: str,
        metadata=None,
        podcaster_speaker: Optional[str] = None,
    ) -> StructuredTranscript:
        """Build a structured transcript from transcription + diarization + metadata."""

        # Map speaker IDs to labels using metadata context
        speaker_labels = {}
        speaker_profiles = []

        # First: extract guest names from metadata
        guest_names = self.extract_guest_names_from_metadata(metadata) if metadata else {}

        # Assign speaker labels
        speaker_labels = self._assign_speaker_labels(
            diarization_result.speaker_segments,
            metadata,
            podcaster_speaker,
            guest_names,
        )

        # Build speaker profiles
        for seg in diarization_result.speaker_segments:
            speaker_id = seg["speaker"]
            if speaker_id not in speaker_profiles:
                speaker_profiles.append(SpeakerProfile(
                    speaker_id=speaker_id,
                    label=speaker_labels[speaker_id],
                    first_appearance=seg["start"],
                ))

        # Merge transcription segments with diarization speaker labels
        merged_segments = []
        for trans_seg in transcription_result.segments:
            speaker_id = self._find_speaker_for_segment(
                trans_seg["start"],
                diarization_result.speaker_segments,
            )

            if speaker_id and speaker_id in speaker_labels:
                merged_segments.append({
                    "start": round(trans_seg["start"], 1),
                    "end": round(trans_seg["end"], 1),
                    "speaker_label": speaker_labels[speaker_id],
                    "text": trans_seg["text"],
                })

        # Save to JSON file in per-video folder
        video_folder = self._get_video_folder(video_title)
        transcript_folder = os.path.join(video_folder, "transcript")
        os.makedirs(transcript_folder, exist_ok=True)

        sanitized_title = self._sanitize_filename(video_title)
        output_path = os.path.join(
            transcript_folder,
            f"{sanitized_title}_transcript.json",
        )

        transcript_data = {
            "video_title": video_title,
            "audio_path": transcription_result.audio_path,
            "language": transcription_result.language,
            "duration": transcription_result.duration,
            "speakers": [
                {
                    "speaker_id": s.speaker_id,
                    "label": s.label,
                    "first_appearance": s.first_appearance,
                }
                for s in speaker_profiles
            ],
            "segments": merged_segments,
            "raw_text": transcription_result.text,
        }

        with open(output_path, "w") as f:
            json.dump(transcript_data, f, indent=2)

        logger.info(f"Transcript saved to: {output_path}")
        logger.info(f"Speakers: {len(speaker_profiles)}, segments: {len(merged_segments)}")

        return StructuredTranscript(
            video_title=video_title,
            audio_path=transcription_result.audio_path,
            language=transcription_result.language,
            duration=transcription_result.duration,
            speakers=speaker_profiles,
            segments=merged_segments,
            raw_text=transcription_result.text,
            output_path=output_path,
        )
