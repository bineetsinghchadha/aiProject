"""Voice resolution for Chatterbox speaker cloning.

Resolves which reference WAV to use for a given target language and optional
speaker ID.  Also provides helpers for loading a diarization sidecar and
mapping transcript segment timestamps to speaker labels.

Directory layout expected under ``pipeline_data/speakers/``::

    speakers/
    ├── default.wav          ← global fallback (any language, unknown speaker)
    ├── es/
    │   ├── default.wav      ← language-specific fallback for Spanish
    │   ├── SPEAKER_00.wav   ← per-speaker voice clone reference
    │   └── SPEAKER_01.wav
    └── fr/
        ├── default.wav
        └── SPEAKER_00.wav

The Chatterbox container mounts ``pipeline_data/speakers/`` at ``/app/voices/``
so the returned relative path (e.g. ``"es/SPEAKER_00.wav"``) can be used
directly in the ``/v1/audio/speech/upload`` API call.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------

def resolve_speaker_wav(
    speakers_dir: Path,
    target_language: str,
    speaker_id: str | None = None,
) -> str:
    """Resolve the reference WAV path for voice cloning.

    Resolution order
    ----------------
    1. ``{lang}/{speaker_id}.wav``  — per-speaker clone (if *speaker_id* given
       and the file exists under *speakers_dir*).
    2. ``{lang}/default.wav``       — language-specific default voice.
    3. ``default.wav``              — global fallback voice.
    4. ``""``                       — empty string if none of the above exist;
       ``ChatterboxClient`` will use the server's built-in default voice.

    Args:
        speakers_dir: Absolute ``Path`` to the speakers directory
            (e.g. ``pipeline_data/speakers``).
        target_language: BCP-47 language code, e.g. ``"es"``, ``"fr"``.
        speaker_id: Diarization label, e.g. ``"SPEAKER_00"``.  Pass ``None``
            (or omit) when no diarization data is available.

    Returns:
        Relative path string suitable for the Chatterbox container,
        e.g. ``"es/SPEAKER_00.wav"`` or ``"es/default.wav"``.
        Returns ``""`` when no reference WAV can be located so the caller
        can fall back to the server's default voice without raising.

    Examples::

        >>> resolve_speaker_wav(Path("/data/speakers"), "es", "SPEAKER_00")
        'es/SPEAKER_00.wav'          # if file exists

        >>> resolve_speaker_wav(Path("/data/speakers"), "es", "SPEAKER_99")
        'es/default.wav'             # speaker-specific missing → lang default

        >>> resolve_speaker_wav(Path("/data/speakers"), "zh")
        'default.wav'                # no zh/ directory → global fallback
    """
    lang = target_language.lower().strip()

    # ── Candidate 1: per-speaker ──────────────────────────────────────────
    if speaker_id:
        speaker_path = speakers_dir / lang / f"{speaker_id}.wav"
        if speaker_path.exists():
            rel = f"{lang}/{speaker_id}.wav"
            logger.debug("[voice] resolved per-speaker WAV: %s", rel)
            return rel

    # ── Candidate 2: language-specific default ────────────────────────────
    lang_default = speakers_dir / lang / "default.wav"
    if lang_default.exists():
        rel = f"{lang}/default.wav"
        logger.debug("[voice] resolved lang-default WAV: %s", rel)
        return rel

    # ── Candidate 3: global fallback ─────────────────────────────────────
    global_default = speakers_dir / "default.wav"
    if global_default.exists():
        logger.debug("[voice] resolved global-default WAV")
        return "default.wav"

    # ── Nothing found ─────────────────────────────────────────────────────
    logger.warning(
        "[voice] no reference WAV found for lang=%s speaker=%s under %s; "
        "Chatterbox will use its built-in default voice",
        lang, speaker_id, speakers_dir,
    )
    return ""


# ---------------------------------------------------------------------------
# Diarization sidecar helpers
# ---------------------------------------------------------------------------

def load_diarization(diarization_path: Path | str) -> list[dict]:
    """Load a diarization sidecar JSON and return its segment list.

    Expected format (written by ``AlignmentService.diarize``)::

        [
            {"start_s": 0.0,  "end_s": 3.2,  "speaker": "SPEAKER_00"},
            {"start_s": 3.5,  "end_s": 7.1,  "speaker": "SPEAKER_01"},
            ...
        ]

    Returns an empty list if the file does not exist or cannot be parsed,
    so callers degrade gracefully to single-voice mode.

    Args:
        diarization_path: Path to the ``.diarize.json`` sidecar file.

    Returns:
        List of diarization segment dicts, or ``[]`` on any error.
    """
    path = Path(diarization_path)
    if not path.exists():
        logger.debug("[voice] diarization sidecar not found at %s", path)
        return []
    try:
        data = json.loads(path.read_text())
        # Accept both a bare list and a dict with a "segments" key
        # (the DiarizeResponse schema wraps segments in an object).
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("segments", [])
        logger.warning("[voice] unexpected diarization format in %s", path)
        return []
    except Exception as exc:
        logger.warning("[voice] failed to load diarization from %s: %s", path, exc)
        return []


def speaker_for_segment(
    segment_start: float,
    segment_end: float,
    diarization: list[dict],
    overlap_threshold: float = 0.3,
) -> str | None:
    """Return the speaker label that best covers a transcript segment.

    We compute how many seconds each speaker's diarization intervals overlap
    with ``[segment_start, segment_end]`` and return the speaker with the
    greatest overlap.  Returns ``None`` when *diarization* is empty or no
    interval overlaps by at least *overlap_threshold* seconds.

    Args:
        segment_start: Start time of the transcript segment (seconds).
        segment_end:   End time of the transcript segment (seconds).
        diarization:   List of ``{start_s, end_s, speaker}`` dicts from
            ``load_diarization()``.
        overlap_threshold: Minimum overlap in seconds to count as a match.
            Prevents very short cross-talk fragments from mis-assigning the
            voice.  Default is 0.3 s.

    Returns:
        Speaker label string (e.g. ``"SPEAKER_00"``) or ``None``.
    """
    if not diarization:
        return None

    overlap_by_speaker: dict[str, float] = {}
    seg_len = segment_end - segment_start

    for d in diarization:
        d_start = d.get("start_s", d.get("start", 0.0))
        d_end   = d.get("end_s",   d.get("end",   0.0))
        speaker = d.get("speaker", "")
        if not speaker:
            continue

        # Overlap = intersection of [segment_start, segment_end] and [d_start, d_end]
        overlap = max(0.0, min(segment_end, d_end) - max(segment_start, d_start))
        if overlap > 0:
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap

    if not overlap_by_speaker:
        return None

    best_speaker, best_overlap = max(overlap_by_speaker.items(), key=lambda kv: kv[1])

    # Require at least overlap_threshold OR 20% of segment length (whichever is
    # smaller) to avoid noise from very short diarization fragments.
    min_required = min(overlap_threshold, seg_len * 0.2)
    if best_overlap < min_required:
        logger.debug(
            "[voice] segment [%.2f, %.2f]: best overlap %.3fs below threshold %.3fs",
            segment_start, segment_end, best_overlap, min_required,
        )
        return None

    return best_speaker


def build_speaker_wav_map(
    diarization: list[dict],
    speakers_dir: Path,
    target_language: str,
) -> dict[str, str]:
    """Pre-resolve a ``{speaker_id: wav_path}`` map for all speakers in *diarization*.

    Calling ``resolve_speaker_wav`` once per speaker (not once per segment)
    avoids redundant filesystem lookups during the hot synthesis loop.

    Args:
        diarization:     Diarization segments from ``load_diarization()``.
        speakers_dir:    Root speakers directory (absolute ``Path``).
        target_language: Language code used for subdirectory lookup.

    Returns:
        Dict mapping each unique speaker label to its resolved WAV path
        (may be ``""`` for speakers with no reference file).
    """
    speakers = {d.get("speaker", "") for d in diarization if d.get("speaker")}
    wav_map: dict[str, str] = {}
    for speaker in speakers:
        wav_map[speaker] = resolve_speaker_wav(speakers_dir, target_language, speaker)
        logger.info("[voice] %s → %s", speaker, wav_map[speaker] or "<server default>")
    return wav_map