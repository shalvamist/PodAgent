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

    def __init__(self, transcript_dir: str = "data/transcripts"):
        self.transcript_dir = transcript_dir
        os.makedirs(transcript_dir, exist_ok=True)

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename."""
        invalid_chars = '<>:\"/\\|?*'
        for c in invalid_chars:
            name = name.replace(c, "_")
        name = name.strip()
        return name[:100]

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
        """Assign speaker labels based on metadata context and diarization order."""
        speaker_labels = {}

        # First: assign known guest names from metadata
        for name, label in guest_names.items():
            # We can't directly map speaker_id to name from diarization alone
            # So we assign based on order: first speaker = podcaster, rest = guests
            pass

        # Then: assign speakers based on order
        for seg in diarization_segments:
            speaker_id = seg["speaker"]
            if speaker_id not in speaker_labels:
                if speaker_id == podcaster_speaker:
                    label = "Podcaster"
                elif metadata and hasattr(metadata, "uploader") and metadata.uploader:
                    # First non-podcaster speaker = channel host/podcaster
                    label = metadata.uploader
                else:
                    # Remaining speakers = guests
                    guest_num = len([s for s in speaker_labels.values()
                                     if s.startswith("Guest")]) + 1
                    label = f"Guest {guest_num}"
                speaker_labels[speaker_id] = label

        logger.info(f"Assigned speaker labels: {speaker_labels}")
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

        # Save to JSON file
        sanitized_title = self._sanitize_filename(video_title)
        output_path = os.path.join(
            self.transcript_dir,
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
