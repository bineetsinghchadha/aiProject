"""HTTP-agnostic service wrapping TTS engine functions."""

import logging
import pathlib
from pathlib import Path
from typing import Any

from api.src.services.tts_engine import text_file_to_speech as tts_text_file_to_speech

logger = logging.getLogger(__name__)


class TTSService:
    """Thin wrapper around the TTS pipeline.

    Accepts *ui_dir*, a pre-loaded *tts_engine*, and an optional
    *speakers_dir* via constructor injection.

    Per-speaker voice selection
    ---------------------------
    When ``speakers_dir`` is supplied and ``text_file_to_speech`` is called
    with a ``diarization_path``, each transcript segment is synthesized using
    the Chatterbox reference voice for the speaker who dominates that segment's
    time window.  The resolution order (per-speaker WAV → lang default →
    global default → server built-in) is handled by
    ``foreign_whispers.voice_resolution.resolve_speaker_wav``.
    """

    def __init__(
        self,
        ui_dir: Path,
        tts_engine: Any,
        speakers_dir: Path | None = None,
    ) -> None:
        self.ui_dir = ui_dir
        self.tts_engine = tts_engine
        # Absolute path to pipeline_data/speakers/ — may be None when running
        # without voice-cloning support (single-voice mode).
        self.speakers_dir: Path | None = speakers_dir

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    def text_file_to_speech(
        self,
        source_path: str,
        output_path: str,
        *,
        alignment: bool | None = None,
        diarization_path: str | None = None,
        target_language: str = "es",
    ) -> None:
        """Generate time-aligned TTS audio from a translated JSON transcript.

        Args:
            source_path:      Path to the translated ``.json`` transcript.
            output_path:      Directory where the output ``.wav`` will be written.
            alignment:        Override ``FW_ALIGNMENT`` env var.
            diarization_path: Path to the ``.diarize.json`` sidecar file.
                              When provided (along with ``self.speakers_dir``),
                              enables per-speaker voice cloning.
            target_language:  BCP-47 language code for voice directory lookup.
        """
        speakers_dir_str: str | None = (
            str(self.speakers_dir) if self.speakers_dir else None
        )

        if diarization_path and not speakers_dir_str:
            logger.warning(
                "[tts_service] diarization_path supplied but speakers_dir is not "
                "configured — per-speaker voice selection will be skipped"
            )

        tts_text_file_to_speech(
            source_path,
            output_path,
            self.tts_engine,
            alignment=alignment,
            diarization_path=diarization_path,
            speakers_dir=speakers_dir_str,
            target_language=target_language,
        )

    # ------------------------------------------------------------------
    # Voice resolution facade
    # ------------------------------------------------------------------

    def resolve_speaker_wav(
        self,
        speaker_id: str | None,
        target_language: str,
    ) -> str:
        """Resolve the reference WAV path for a given speaker and language.

        Delegates to ``foreign_whispers.voice_resolution.resolve_speaker_wav``.
        Returns ``""`` when ``speakers_dir`` is not configured.

        Args:
            speaker_id:      Diarization label (e.g. ``"SPEAKER_00"``), or
                             ``None`` for the language/global default.
            target_language: BCP-47 language code (e.g. ``"es"``).

        Returns:
            Relative WAV path string suitable for Chatterbox, or ``""``.
        """
        if not self.speakers_dir:
            logger.debug("[tts_service] speakers_dir not set — returning empty wav path")
            return ""
        from foreign_whispers.voice_resolution import resolve_speaker_wav
        return resolve_speaker_wav(self.speakers_dir, target_language, speaker_id)

    def list_available_speakers(self, target_language: str) -> list[str]:
        """Return speaker IDs that have a reference WAV for *target_language*.

        Scans ``speakers_dir/{target_language}/`` for ``*.wav`` files,
        excluding ``default.wav``.  Returns an empty list when ``speakers_dir``
        is not configured or the language directory does not exist.

        Args:
            target_language: BCP-47 language code (e.g. ``"es"``).

        Returns:
            Sorted list of speaker ID strings (stems of the WAV files found).
        """
        if not self.speakers_dir:
            return []
        lang_dir = self.speakers_dir / target_language.lower()
        if not lang_dir.is_dir():
            return []
        return sorted(
            f.stem
            for f in lang_dir.glob("*.wav")
            if f.stem.lower() != "default"
        )

    # ------------------------------------------------------------------
    # Alignment facade
    # ------------------------------------------------------------------

    def compute_alignment(
        self,
        en_transcript: dict,
        es_transcript: dict,
        silence_regions: list[dict],
        max_stretch: float = 1.4,
    ) -> list:
        """Run global alignment over EN and ES transcripts.

        Returns list[AlignedSegment].  Combines compute_segment_metrics and
        global_align into a single facade call for use by the align router.
        """
        from foreign_whispers.alignment import compute_segment_metrics, global_align
        metrics = compute_segment_metrics(en_transcript, es_transcript)
        return global_align(metrics, silence_regions, max_stretch)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def title_for_video_id(video_id: str, search_dir: pathlib.Path) -> str | None:
        """Find a title by scanning *search_dir* for JSON files."""
        for f in search_dir.glob("*.json"):
            return f.stem
        return None