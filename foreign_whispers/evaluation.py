"""Clip-level alignment quality metrics — multi-dimensional scorecard.

This module provides two levels of evaluation:

``clip_evaluation_report``
    Lightweight, always-available report (no optional deps).  Returns the
    five timing metrics from the original assignment plus three new ones:
    speaking-rate variance, segment-level overlap count, and a composite
    timing score.

``clip_quality_scorecard``
    Full multi-dimensional scorecard.  Four dimensions:

    1. **Timing accuracy** — how closely predicted TTS durations match the
       source windows (MAE, % severe stretch, drift, overlap).
    2. **Intelligibility** — optional STT round-trip: synthesise each segment,
       transcribe with Whisper, compute word-error-rate against the source.
    3. **Semantic fidelity** — optional embedding cosine similarity between
       source and translated text using ``sentence-transformers``.
    4. **Naturalness** — speaking-rate variance across segments; high variance
       means some segments are drastically sped up while others are slow.

    Optional dimensions degrade gracefully: if the required dep is absent the
    field is ``None`` with a ``skipped_reason`` explanation.

``compare_scorecards``
    Convenience function for notebook cells: runs both ``global_align`` and
    ``global_align_dp`` and returns a side-by-side scorecard comparison.
"""
from __future__ import annotations

import math
import statistics as _stats
from typing import Any

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _speaking_rate_variance(aligned: list[AlignedSegment]) -> float:
    """Return the variance of per-segment stretch factors.

    Low variance ≈ uniform pacing; high variance ≈ some segments are
    drastically sped up or slowed down.
    """
    if len(aligned) < 2:
        return 0.0
    rates = [a.stretch_factor for a in aligned]
    return round(_stats.variance(rates), 4)


def _count_overlaps(aligned: list[AlignedSegment]) -> int:
    """Count adjacent segment pairs where the second starts before the first ends."""
    return sum(
        1 for a, b in zip(aligned, aligned[1:])
        if b.scheduled_start < a.scheduled_end - 0.01  # 10ms tolerance
    )


def _timing_score(
    mean_err: float,
    pct_severe: float,
    drift: float,
    n_overlaps: int,
) -> float:
    """Composite 0–100 timing score (higher = better).

    Penalty formula (each term contributes independently):

    - mean_err  : -8 pts per 0.1 s above 0.05 s
    - pct_severe: -1.5 pts per % of severely-stretched segments
    - drift     : -4 pts per second of total drift
    - overlaps  : -10 pts per overlapping segment pair

    Score is clamped to [0, 100].
    """
    score = 100.0
    score -= max(0.0, (mean_err - 0.05) / 0.1) * 8.0
    score -= pct_severe * 1.5
    score -= abs(drift) * 4.0
    score -= n_overlaps * 10.0
    return round(max(0.0, min(100.0, score)), 1)


# ---------------------------------------------------------------------------
# Tier 1: lightweight timing report (original + extended)
# ---------------------------------------------------------------------------

def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    This is the lightweight always-available report.  No optional deps needed.

    Keys (original five, preserved for backwards compatibility):
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s|.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end schedule drift from gap-shifts.

    New keys (added in scorecard extension):
        n_overlaps: Number of adjacent segment pairs that overlap in time.
        speaking_rate_variance: Variance of per-segment stretch factors.
        timing_score: Composite 0–100 score (higher = better).
        n_accept: Segments accepted without modification.
        n_mild_stretch: Segments requiring mild time-stretch only.
        n_fail: Segments with no feasible fix.

    Args:
        metrics: Per-segment timing metrics (from ``compute_segment_metrics``).
        aligned: Scheduled segments (from ``global_align`` or ``global_align_dp``).

    Returns:
        Dict of alignment quality metrics.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
            "n_overlaps":                0,
            "speaking_rate_variance":    0.0,
            "timing_score":              100.0,
            "n_accept":                  0,
            "n_mild_stretch":            0,
            "n_fail":                    0,
        }

    errors     = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe   = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted  = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry    = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    n_accept   = sum(1 for a in aligned if a.action == AlignAction.ACCEPT)
    n_mild     = sum(1 for a in aligned if a.action == AlignAction.MILD_STRETCH)
    n_fail     = sum(1 for a in aligned if a.action == AlignAction.FAIL)
    drift      = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )
    n_overlaps = _count_overlaps(aligned)
    sr_var     = _speaking_rate_variance(aligned)
    mean_err   = round(_stats.mean(errors), 3)
    pct_severe = round(100 * n_severe / max(len(metrics), 1), 1)

    return {
        "mean_abs_duration_error_s": mean_err,
        "pct_severe_stretch":        pct_severe,
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
        "n_overlaps":                n_overlaps,
        "speaking_rate_variance":    sr_var,
        "timing_score":              _timing_score(mean_err, pct_severe, drift, n_overlaps),
        "n_accept":                  n_accept,
        "n_mild_stretch":            n_mild,
        "n_fail":                    n_fail,
    }


# ---------------------------------------------------------------------------
# Tier 2: full multi-dimensional scorecard
# ---------------------------------------------------------------------------

def _dimension_timing(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Dimension 1: timing accuracy."""
    base = clip_evaluation_report(metrics, aligned)
    return {
        "score":                     base["timing_score"],
        "mean_abs_duration_error_s": base["mean_abs_duration_error_s"],
        "pct_severe_stretch":        base["pct_severe_stretch"],
        "total_cumulative_drift_s":  base["total_cumulative_drift_s"],
        "n_overlaps":                base["n_overlaps"],
        "n_gap_shifts":              base["n_gap_shifts"],
        "n_translation_retries":     base["n_translation_retries"],
        "n_accept":                  base["n_accept"],
        "n_mild_stretch":            base["n_mild_stretch"],
        "n_fail":                    base["n_fail"],
    }


def _dimension_intelligibility(
    aligned:     list[AlignedSegment],
    audio_dir:   str | None,
    whisper_model: Any | None,
) -> dict:
    """Dimension 2: intelligibility via STT round-trip WER.

    Requires:
    - ``audio_dir``: directory containing per-segment WAV files named
      ``seg_{index}.wav``.
    - ``whisper_model``: a loaded Whisper model instance with a
      ``transcribe(path) -> dict`` method.

    Word Error Rate is computed as::

        WER = (S + D + I) / N

    where S = substitutions, D = deletions, I = insertions, N = reference
    word count.  We use a simple token-set approach (not the full DP edit
    distance) which is fast and sufficient for relative comparisons.

    Returns a dict with ``score`` (0–100, higher = better), ``mean_wer``,
    and ``skipped_reason`` (non-None when the dimension was skipped).
    """
    if audio_dir is None or whisper_model is None:
        return {
            "score":         None,
            "mean_wer":      None,
            "skipped_reason": "audio_dir or whisper_model not provided",
        }

    import pathlib

    def _simple_wer(reference: str, hypothesis: str) -> float:
        """Token-set WER approximation."""
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not ref_tokens:
            return 0.0
        ref_set = set(ref_tokens)
        hyp_set = set(hyp_tokens)
        correct = len(ref_set & hyp_set)
        errors  = len(ref_set) - correct + len(hyp_set - ref_set)
        return errors / len(ref_tokens)

    wers = []
    for seg in aligned:
        wav_path = pathlib.Path(audio_dir) / f"seg_{seg.index}.wav"
        if not wav_path.exists():
            continue
        try:
            result   = whisper_model.transcribe(str(wav_path))
            hyp_text = result.get("text", "").strip()
            wer      = _simple_wer(seg.text, hyp_text)
            wers.append(wer)
        except Exception:
            continue  # skip failed segments silently

    if not wers:
        return {
            "score":         None,
            "mean_wer":      None,
            "skipped_reason": "no segment WAV files found or all transcriptions failed",
        }

    mean_wer = round(_stats.mean(wers), 4)
    # Score: 100 at WER=0, 0 at WER=1
    score = round(max(0.0, (1.0 - mean_wer) * 100), 1)
    return {
        "score":          score,
        "mean_wer":       mean_wer,
        "n_segments_evaluated": len(wers),
        "skipped_reason": None,
    }


def _dimension_semantic_fidelity(
    metrics: list[SegmentMetrics],
) -> dict:
    """Dimension 3: semantic fidelity via embedding cosine similarity.

    Uses ``sentence-transformers`` (``paraphrase-multilingual-MiniLM-L12-v2``)
    to encode both source and translated text and measure their cosine
    similarity.  Mean similarity across all segments is returned.

    Requires: ``pip install sentence-transformers``

    Returns a dict with ``score`` (0–100), ``mean_cosine_similarity``, and
    ``skipped_reason``.
    """
    try:
        from sentence_transformers import SentenceTransformer, util as st_util
    except ImportError:
        return {
            "score":                  None,
            "mean_cosine_similarity": None,
            "skipped_reason":         "sentence-transformers not installed",
        }

    if not metrics:
        return {
            "score":                  100.0,
            "mean_cosine_similarity": 1.0,
            "skipped_reason":         None,
        }

    try:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        src_texts = [m.source_text     for m in metrics]
        tgt_texts = [m.translated_text for m in metrics]

        src_emb = model.encode(src_texts, convert_to_tensor=True, show_progress_bar=False)
        tgt_emb = model.encode(tgt_texts, convert_to_tensor=True, show_progress_bar=False)

        # Per-segment cosine similarity
        sims = [
            float(st_util.cos_sim(s, t).item())
            for s, t in zip(src_emb, tgt_emb)
        ]
        mean_sim = round(_stats.mean(sims), 4)
        score    = round(max(0.0, min(100.0, mean_sim * 100)), 1)
        return {
            "score":                  score,
            "mean_cosine_similarity": mean_sim,
            "min_cosine_similarity":  round(min(sims), 4),
            "skipped_reason":         None,
        }
    except Exception as exc:
        return {
            "score":                  None,
            "mean_cosine_similarity": None,
            "skipped_reason":         f"embedding computation failed: {exc}",
        }


def _dimension_naturalness(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Dimension 4: naturalness via speaking-rate variance and pace distribution.

    A natural dub has consistent pacing: stretch factors close to 1.0 and
    low variance.  We report:

    - ``mean_stretch_factor``: ideal is 1.0.
    - ``speaking_rate_variance``: lower is more natural (ideal < 0.02).
    - ``pct_within_natural_range``: % of segments with stretch in [0.9, 1.2].
    - ``score``: 0–100 composite (higher = more natural pacing).

    Score formula::

        base = 100
        base -= variance * 500          # penalise uneven pacing
        base -= |mean_stretch - 1| * 40  # penalise systematic speed-up/slowdown
        base -= (100 - pct_natural) * 0.3
    """
    if not aligned:
        return {
            "score":                  100.0,
            "mean_stretch_factor":    1.0,
            "speaking_rate_variance": 0.0,
            "pct_within_natural_range": 100.0,
            "skipped_reason":         None,
        }

    stretches = [a.stretch_factor for a in aligned]
    mean_sf   = round(_stats.mean(stretches), 4)
    var_sf    = _speaking_rate_variance(aligned)
    pct_nat   = round(
        100 * sum(1 for s in stretches if 0.9 <= s <= 1.2) / len(stretches), 1
    )

    score = 100.0
    score -= var_sf * 500
    score -= abs(mean_sf - 1.0) * 40
    score -= (100 - pct_nat) * 0.3
    score = round(max(0.0, min(100.0, score)), 1)

    return {
        "score":                    score,
        "mean_stretch_factor":      mean_sf,
        "speaking_rate_variance":   var_sf,
        "pct_within_natural_range": pct_nat,
        "skipped_reason":           None,
    }


def clip_quality_scorecard(
    metrics:        list[SegmentMetrics],
    aligned:        list[AlignedSegment],
    *,
    audio_dir:      str | None = None,
    whisper_model:  Any | None = None,
) -> dict:
    """Full multi-dimensional quality scorecard for one dubbed clip.

    Runs all four evaluation dimensions and returns a composite dict.
    Dimensions whose optional dependencies are missing are included with
    ``score: None`` and a ``skipped_reason`` explanation.

    Dimensions
    ----------
    1. **timing**       — always computed; no external deps.
    2. **intelligibility** — requires ``audio_dir`` + ``whisper_model``.
    3. **semantic_fidelity** — requires ``sentence-transformers``.
    4. **naturalness**  — always computed; no external deps.

    Composite score
    ---------------
    The composite is the mean of all non-None dimension scores, weighted::

        composite = (2·timing + 1·intelligibility + 2·semantic + 1·naturalness)
                    / sum_of_weights_for_available_dims

    Args:
        metrics:       Per-segment timing metrics.
        aligned:       Scheduled segments.
        audio_dir:     Directory of per-segment WAV files (for intelligibility).
        whisper_model: Loaded Whisper model (for intelligibility).

    Returns:
        Dict with keys ``"composite_score"``, ``"timing"``,
        ``"intelligibility"``, ``"semantic_fidelity"``, ``"naturalness"``,
        ``"n_segments"``.
    """
    dim_timing   = _dimension_timing(metrics, aligned)
    dim_intel    = _dimension_intelligibility(aligned, audio_dir, whisper_model)
    dim_semantic = _dimension_semantic_fidelity(metrics)
    dim_natural  = _dimension_naturalness(metrics, aligned)

    # Weighted composite (only available dimensions contribute)
    weights = {
        "timing":            2.0,
        "intelligibility":   1.0,
        "semantic_fidelity": 2.0,
        "naturalness":       1.0,
    }
    scores = {
        "timing":            dim_timing["score"],
        "intelligibility":   dim_intel["score"],
        "semantic_fidelity": dim_semantic["score"],
        "naturalness":       dim_natural["score"],
    }

    total_weight = sum(w for k, w in weights.items() if scores[k] is not None)
    if total_weight > 0:
        composite = sum(
            scores[k] * weights[k]
            for k in weights
            if scores[k] is not None
        ) / total_weight
    else:
        composite = None

    return {
        "composite_score":    round(composite, 1) if composite is not None else None,
        "n_segments":         len(metrics),
        "timing":             dim_timing,
        "intelligibility":    dim_intel,
        "semantic_fidelity":  dim_semantic,
        "naturalness":        dim_natural,
    }


# ---------------------------------------------------------------------------
# Comparison helper for notebooks
# ---------------------------------------------------------------------------

def compare_scorecards(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> dict:
    """Run greedy and DP aligners, return side-by-side scorecards.

    Intended for notebook cells::

        results = compare_scorecards(metrics, silence_regions)
        print(results["improvement"])

    Returns:
        Dict with keys ``"greedy"``, ``"dp"``, and ``"improvement"`` (a
        human-readable summary of how much the DP scheduler improved things).
    """
    from foreign_whispers.alignment import global_align, global_align_dp

    greedy_aligned = global_align(metrics, silence_regions, max_stretch)
    dp_aligned     = global_align_dp(metrics, silence_regions, max_stretch)

    greedy_report  = clip_evaluation_report(metrics, greedy_aligned)
    dp_report      = clip_evaluation_report(metrics, dp_aligned)

    drift_delta   = greedy_report["total_cumulative_drift_s"]  - dp_report["total_cumulative_drift_s"]
    severe_delta  = greedy_report["pct_severe_stretch"]        - dp_report["pct_severe_stretch"]
    overlap_delta = greedy_report["n_overlaps"]                - dp_report["n_overlaps"]
    score_delta   = dp_report["timing_score"]                  - greedy_report["timing_score"]

    improvement = (
        f"DP vs Greedy: timing_score {greedy_report['timing_score']:.1f} → "
        f"{dp_report['timing_score']:.1f} ({score_delta:+.1f} pts); "
        f"drift {greedy_report['total_cumulative_drift_s']:.2f}s → "
        f"{dp_report['total_cumulative_drift_s']:.2f}s ({-drift_delta:+.2f}s); "
        f"severe_stretch {greedy_report['pct_severe_stretch']:.1f}% → "
        f"{dp_report['pct_severe_stretch']:.1f}% ({-severe_delta:+.1f}pp); "
        f"overlaps {greedy_report['n_overlaps']} → {dp_report['n_overlaps']}"
    )

    return {
        "greedy":      greedy_report,
        "dp":          dp_report,
        "improvement": improvement,
    }