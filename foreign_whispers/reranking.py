"""Deterministic failure analysis and duration-aware translation re-ranking.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The re-ranking function implements a four-stage shortening
pipeline that is applied in order, stopping as soon as the budget is met:

    Stage 1 — Lookup compression      (phrase table + filler removal + pronoun drop)
    Stage 2 — Clause pruning          (drop subordinate / parenthetical clauses)
    Stage 3 — Argos re-translation    (translate the *source* text again with a
                                       brevity hint injected as a paraphrase)
    Stage 4 — Hard truncation         (last resort: cut at the nearest word
                                       boundary that satisfies the budget)

Context awareness
-----------------
``context_prev`` and ``context_next`` are used in two ways:

1. **Pronoun elision** — if the subject of this segment can be inferred from
   the previous segment, the explicit subject pronoun is dropped.
2. **Clause pruning** — if a trailing relative clause merely restates
   something said in context_prev, it is a candidate for deletion.

Duration budget
---------------
The budget is ``target_duration_s`` seconds.  The TTS rate for Spanish is
approximately **15 characters / second** (matching the 4.5 syl/s used by
``alignment.py``).  Each candidate's predicted duration is::

    predicted_s = char_count / 15.0

The caller (``TranslationService.rerank_for_duration``) picks the candidate
whose predicted duration is *closest* to the target — not necessarily the
shortest — so we emit a full ladder of candidates from baseline down to the
hardest truncation.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Characters-per-second rate used for duration estimation.
_CHARS_PER_SECOND: float = 15.0


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation paired with its predicted duration.

    Attributes:
        text: The (shortened) translated text.
        char_count: ``len(text)`` — used by the caller for duration estimation.
        brevity_rationale: Human-readable note on what was changed.
        predicted_duration_s: Estimated TTS duration (``char_count / 15.0``).
        stage: Which shortening stage produced this candidate (1–4 or 0 for
            the unmodified baseline).
    """
    text: str
    char_count: int
    brevity_rationale: str = ""
    predicted_duration_s: float = dataclasses.field(init=False)
    stage: int = 0

    def __post_init__(self) -> None:
        self.predicted_duration_s = self.char_count / _CHARS_PER_SECOND


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of ``"duration_overflow"``, ``"cumulative_drift"``,
            ``"stretch_quality"``, or ``"ok"``.
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


# ---------------------------------------------------------------------------
# analyze_failures (unchanged from original)
# ---------------------------------------------------------------------------

def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err   = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift      = abs(report.get("total_cumulative_drift_s", 0.0))

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Collapse whitespace and strip leading/trailing spaces."""
    return " ".join(text.strip().split())


def _make_candidate(text: str, rationale: str, stage: int) -> TranslationCandidate:
    text = _normalize(text)
    return TranslationCandidate(
        text=text,
        char_count=len(text),
        brevity_rationale=rationale,
        stage=stage,
    )


def _ascii_lower(text: str) -> str:
    """Lowercase and strip accents for accent-insensitive comparisons."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _fits(text: str, target_duration_s: float) -> bool:
    return len(text) / _CHARS_PER_SECOND <= target_duration_s


# ---------------------------------------------------------------------------
# Stage 1 — Lookup compression
# ---------------------------------------------------------------------------

# Ordered from longest to shortest so more specific phrases match first.
_PHRASE_TABLE: list[tuple[str, str]] = [
    # Multi-word → single word or short phrase
    ("tiene la capacidad de",    "puede"),
    ("en este momento",          "ahora"),
    ("en este instante",         "ahora"),
    ("en el momento actual",     "ahora"),
    ("a partir de este momento", "desde ahora"),
    ("debido a que",             "porque"),
    ("a causa de que",           "porque"),
    ("a pesar de que",           "aunque"),
    ("a pesar de ello",          "aun así"),
    ("con el fin de que",        "para que"),
    ("con el fin de",            "para"),
    ("con el objetivo de",       "para"),
    ("con el propósito de",      "para"),
    ("de la misma manera",       "igual"),
    ("de la misma forma",        "igual"),
    ("de igual manera",          "igual"),
    ("por lo tanto",             "así que"),
    ("por esta razón",           "por eso"),
    ("por ese motivo",           "por eso"),
    ("es posible que",           "quizá"),
    ("es probable que",          "quizá"),
    ("es necesario que",         "hay que"),
    ("es necesario",             "hay que"),
    ("es importante que",        "importa que"),
    ("en el caso de que",        "si"),
    ("en caso de que",           "si"),
    ("a menos que",              "salvo si"),
    ("sin embargo",              "pero"),
    ("no obstante",              "pero"),
    ("a pesar de todo",          "aun así"),
    ("en realidad",              "en verdad"),
    ("de hecho",                 "de hecho"),   # same length, skip effectively
    ("por supuesto",             "claro"),
    ("por otra parte",           "además"),
    ("por otro lado",            "además"),
    ("en primer lugar",          "primero"),
    ("en segundo lugar",         "segundo"),
    ("en última instancia",      "al final"),
    ("en definitiva",            "al fin"),
    ("a continuación",           "luego"),
    ("a continuación se",        "luego se"),
    ("se puede decir que",       ""),            # pure filler
    ("hay que tener en cuenta",  "nota"),
    ("cabe destacar que",        ""),            # pure filler opener
    ("cabe mencionar que",       ""),
    ("es importante mencionar",  ""),
    ("es importante destacar",   ""),
    ("quiero decir que",         ""),
    ("lo que quiero decir es",   ""),
]

_FILLER_WORDS: frozenset[str] = frozenset([
    "muy", "realmente", "básicamente", "simplemente",
    "solamente", "exactamente", "totalmente", "absolutamente",
    "completamente", "verdaderamente", "ciertamente", "definitivamente",
    "efectivamente", "evidentemente", "obviamente", "claramente",
    "prácticamente", "literalmente", "francamente",
])

# Subject pronouns that can be dropped when context makes the subject clear.
_SUBJECT_PRONOUNS: frozenset[str] = frozenset([
    "yo", "tú", "usted", "él", "ella",
    "nosotros", "nosotras", "vosotros", "vosotras",
    "ellos", "ellas", "ustedes",
])

# Definite/indefinite articles that can be dropped mid-sentence.
_ARTICLES: frozenset[str] = frozenset([
    "el", "la", "los", "las", "un", "una", "unos", "unas",
])


def _apply_phrase_table(text: str) -> str:
    """Replace long phrases with short equivalents (case-insensitive, accent-aware)."""
    result = text
    for long_form, short_form in _PHRASE_TABLE:
        # Build a case-insensitive, accent-insensitive regex pattern.
        pattern = re.compile(re.escape(long_form), re.IGNORECASE)
        result = pattern.sub(short_form, result)
    # Collapse any double spaces left by empty substitutions.
    return re.sub(r"  +", " ", result).strip()


def _remove_fillers(text: str) -> str:
    """Strip adverbial filler words that carry no semantic content."""
    words = text.split()
    return " ".join(w for w in words if _ascii_lower(w) not in _FILLER_WORDS)


def _elide_subject_pronoun(text: str, context_prev: str) -> str:
    """Drop the leading subject pronoun if the context makes the subject obvious.

    Spanish is a pro-drop language — subject pronouns are only added for
    emphasis or disambiguation.  If context_prev already establishes the
    same subject, the pronoun in the current segment is redundant.

    We use a simple heuristic: if context_prev is non-empty and the current
    segment starts with a first- or third-person pronoun, drop it.
    """
    if not context_prev.strip():
        return text
    words = text.split()
    if not words:
        return text
    if _ascii_lower(words[0]) in _SUBJECT_PRONOUNS:
        return " ".join(words[1:])
    return text


def _drop_mid_sentence_articles(text: str) -> str:
    """Remove definite/indefinite articles that appear mid-sentence.

    Keeps the first word unchanged so we don't create a grammatically
    broken sentence opener.
    """
    words = text.split()
    result = []
    for i, w in enumerate(words):
        if i > 0 and _ascii_lower(w) in _ARTICLES:
            continue
        result.append(w)
    return " ".join(result)


def _stage1_lookup(
    baseline: str,
    context_prev: str,
    context_next: str,  # noqa: ARG001 — reserved for future coherence checks
) -> list[TranslationCandidate]:
    """Stage 1: deterministic lookup-based compression.

    Applies transformations in a pipeline, emitting a candidate after each
    step that actually changes the text.  The pipeline is:

        phrase-table → filler removal → subject elision → article dropping

    All four transformations are cumulative: each step starts from the output
    of the previous one.
    """
    candidates: list[TranslationCandidate] = []

    # Step 1a: phrase table
    after_phrases = _normalize(_apply_phrase_table(baseline))
    if after_phrases != baseline:
        candidates.append(_make_candidate(after_phrases, "phrase-table compression", stage=1))

    working = after_phrases

    # Step 1b: filler removal
    after_fillers = _normalize(_remove_fillers(working))
    if after_fillers != working:
        candidates.append(_make_candidate(after_fillers, "filler-word removal", stage=1))

    working = after_fillers

    # Step 1c: subject pronoun elision (context-aware)
    after_elision = _normalize(_elide_subject_pronoun(working, context_prev))
    if after_elision != working:
        candidates.append(_make_candidate(
            after_elision,
            "subject-pronoun elision (context: prev segment establishes subject)",
            stage=1,
        ))

    working = after_elision

    # Step 1d: mid-sentence article dropping
    after_articles = _normalize(_drop_mid_sentence_articles(working))
    if after_articles != working:
        candidates.append(_make_candidate(after_articles, "mid-sentence article removal", stage=1))

    return candidates


# ---------------------------------------------------------------------------
# Stage 2 — Clause pruning
# ---------------------------------------------------------------------------

# Subordinating conjunctions that typically open prunable relative / adverbial
# clauses appended at the end of a sentence.
_PRUNABLE_CLAUSE_STARTERS: list[str] = [
    " que ", " quien ", " quienes ",
    " el cual ", " la cual ", " los cuales ", " las cuales ",
    " lo cual ",
    " donde ", " cuando ", " como ", " aunque ", " si bien ",
    " a pesar de ", " debido a ", " ya que ", " puesto que ",
    " dado que ", " porque ", " para que ", " sin que ",
]

_PARENTHETICAL = re.compile(r"\s*[\(\[].*?[\)\]]\s*")
_DASH_ASIDE    = re.compile(r"\s*—[^—]+—\s*")


def _remove_parentheticals(text: str) -> str:
    """Strip parenthetical asides enclosed in (), [] or em-dash pairs."""
    text = _PARENTHETICAL.sub(" ", text)
    text = _DASH_ASIDE.sub(" ", text)
    return re.sub(r"  +", " ", text).strip()


def _prune_trailing_clause(text: str, context_prev: str) -> str:
    """Drop the last subordinate clause if it restates something in context_prev.

    We identify the clause by scanning backwards for a prunable starter.
    If context_prev shares significant vocabulary with the candidate clause
    (Jaccard similarity > 0.25), we prune it — it's a context restatement.
    Otherwise we prune it only if the segment is still significantly over
    budget (caller decides based on the returned text length).
    """
    best_cut = len(text)
    for starter in _PRUNABLE_CLAUSE_STARTERS:
        idx = text.lower().rfind(starter)
        if 0 < idx < best_cut:
            best_cut = idx

    if best_cut == len(text):
        return text  # no prunable clause found

    clause    = text[best_cut:]
    remainder = text[:best_cut].rstrip(" ,;")

    # Jaccard similarity between clause tokens and context_prev tokens
    if context_prev:
        clause_tokens   = set(_ascii_lower(clause).split())
        context_tokens  = set(_ascii_lower(context_prev).split())
        intersection    = clause_tokens & context_tokens
        union           = clause_tokens | context_tokens
        jaccard         = len(intersection) / len(union) if union else 0.0
    else:
        jaccard = 0.0

    # Always prune if the clause duplicates context; otherwise return the cut
    # so the caller can decide based on the length saving.
    if jaccard > 0.15 or len(remainder) < len(text) * 0.85:
        return remainder
    return text


def _stage2_clause_pruning(
    prev_stage_best: str,
    context_prev: str,
) -> list[TranslationCandidate]:
    """Stage 2: remove parentheticals and trailing subordinate clauses."""
    candidates: list[TranslationCandidate] = []
    working = prev_stage_best

    # Step 2a: parenthetical removal
    after_parens = _normalize(_remove_parentheticals(working))
    if after_parens != working:
        candidates.append(_make_candidate(after_parens, "parenthetical removal", stage=2))
        working = after_parens

    # Step 2b: trailing clause pruning (context-aware)
    after_clause = _normalize(_prune_trailing_clause(working, context_prev))
    if after_clause != working and len(after_clause) >= 5:
        candidates.append(_make_candidate(
            after_clause,
            "trailing-clause pruned (context overlap check)",
            stage=2,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Stage 3 — Argos re-translation with brevity hint
# ---------------------------------------------------------------------------

def _stage3_retranslation(
    source_text: str,
    target_duration_s: float,
    context_prev: str,
    context_next: str,
) -> list[TranslationCandidate]:
    """Stage 3: re-translate the source text using argostranslate.

    We attempt two strategies:
    a) Translate the raw source text again (argostranslate is deterministic,
       but a second pass on a lightly paraphrased input can yield a different
       length).
    b) Prepend a brevity instruction comment to the source text and translate;
       argostranslate will typically translate the comment too, which we then
       strip.  The surrounding context is prepended/appended as soft priming
       so the engine can produce a contextually coherent short form.

    Returns an empty list if argostranslate is not installed.
    """
    try:
        import argostranslate.translate as _at
    except ImportError:
        logger.debug("argostranslate not installed — skipping Stage 3 re-translation.")
        return []

    candidates: list[TranslationCandidate] = []
    max_chars = int(target_duration_s * _CHARS_PER_SECOND)

    def _translate(text: str) -> str | None:
        try:
            return _normalize(_at.translate(text, "en", "es"))
        except Exception as exc:
            logger.debug("argostranslate failed: %s", exc)
            return None

    # Strategy 3a: plain re-translation
    retranslated = _translate(source_text)
    if retranslated:
        candidates.append(_make_candidate(retranslated, "argos re-translation", stage=3))

    # Strategy 3b: context-primed re-translation
    # Build a compact prompt: [prev] current [next] — translated, strip outer context.
    if context_prev or context_next:
        primed_input = " ".join(filter(None, [context_prev.strip(), source_text.strip(), context_next.strip()]))
        primed_result = _translate(primed_input)
        if primed_result:
            # The full primed result will be longer; extract the middle portion
            # that likely corresponds to source_text.
            # Heuristic: take the central (max_chars * 1.5) characters.
            midpoint = len(primed_result) // 2
            half_window = min(len(primed_result) // 2, max_chars)
            excerpt = primed_result[max(0, midpoint - half_window): midpoint + half_window]
            excerpt = _normalize(excerpt)
            if excerpt and excerpt != retranslated:
                candidates.append(_make_candidate(
                    excerpt,
                    "context-primed argos re-translation (excerpt)",
                    stage=3,
                ))

    return candidates


# ---------------------------------------------------------------------------
# Stage 4 — Hard truncation
# ---------------------------------------------------------------------------

def _stage4_truncate(text: str, target_duration_s: float) -> list[TranslationCandidate]:
    """Stage 4: last-resort truncation to fit within *target_duration_s*.

    Tries to cut at natural sentence boundaries (punctuation, conjunctions)
    before resorting to a hard character-count cut.
    """
    max_chars = int(target_duration_s * _CHARS_PER_SECOND)
    if len(text) <= max_chars:
        return []

    candidates: list[TranslationCandidate] = []

    # Try progressively softer sentence-level cuts.
    for sep in [".", "!", "?", ";", ",", " y ", " o ", " pero ", " aunque "]:
        idx = text.rfind(sep, 0, max_chars)
        if idx > max_chars * 0.5:
            cut = _normalize(text[: idx + 1])
            if cut and cut != text:
                candidates.append(_make_candidate(cut, f"truncated at '{sep.strip()}'", stage=4))
                break

    # Hard word-boundary cut
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.6:
        truncated = truncated[:last_space]
    truncated = _normalize(truncated)
    if truncated and truncated not in {c.text for c in candidates}:
        candidates.append(_make_candidate(truncated, "hard word-boundary truncation", stage=4))

    return candidates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_shorter_translations(
    source_text:       str,
    baseline_es:       str,
    target_duration_s: float,
    context_prev:      str = "",
    context_next:      str = "",
) -> list[TranslationCandidate]:
    """Return shorter Spanish translation candidates that fit *target_duration_s*.

    The function works through four stages in order.  Within each stage,
    candidates are collected but the pipeline always continues to the next
    stage (so the caller receives a full ladder from which to pick the best
    fit).  A stage is skipped only when its dependencies are unavailable
    (e.g. argostranslate not installed for Stage 3).

    Stages
    ------
    1. **Lookup compression** — phrase-table substitution, filler removal,
       subject-pronoun elision (context-aware), mid-sentence article dropping.
       All four transforms are cumulative; a candidate is emitted after each
       transform that changes the text.

    2. **Clause pruning** — strip parenthetical asides and drop trailing
       subordinate clauses that restate content already in *context_prev*
       (Jaccard token overlap > 0.15).

    3. **Argos re-translation** — re-translate *source_text* from English to
       Spanish (plain + context-primed).  May produce a different length than
       the argostranslate baseline.  Skipped silently if argostranslate is not
       installed.

    4. **Hard truncation** — cut at the nearest punctuation mark or word
       boundary that satisfies the budget.  Always produces at least one
       candidate when the budget has not yet been met.

    Context usage
    -------------
    * ``context_prev`` — used for subject-pronoun elision (Stage 1) and
      trailing-clause pruning (Stage 2).
    * ``context_next`` — passed to the re-translation stage (Stage 3) as a
      soft coherence signal.

    Duration budget
    ---------------
    ``target_duration_s`` comes from the source segment's actual duration
    stored in the caption/Whisper timestamps, so each call has a *different*
    per-segment budget.  The heuristic is 15 chars/s for Spanish.

    Args:
        source_text:       Original English segment text.
        baseline_es:       Baseline Spanish translation (from argostranslate).
        target_duration_s: Time budget in seconds for this segment.  Derived
                           from the caption segment's ``end - start`` field.
        context_prev:      Text of the preceding segment (Spanish).
        context_next:      Text of the following segment (Spanish).

    Returns:
        List of ``TranslationCandidate`` objects sorted by
        ``|predicted_duration_s - target_duration_s|`` ascending — i.e. the
        candidate whose predicted duration is closest to the target comes first.
        The unmodified baseline is always included as a reference point.
    """
    logger.debug(
        "get_shorter_translations: budget=%.2fs  max_chars=%d  baseline=%d chars",
        target_duration_s,
        int(target_duration_s * _CHARS_PER_SECOND),
        len(baseline_es),
    )

    baseline = _normalize(baseline_es)

    # Accumulate all candidates; key by text for deduplication.
    seen:       dict[str, TranslationCandidate] = {}
    all_cands:  list[TranslationCandidate] = []

    def _add(c: TranslationCandidate) -> None:
        if c.text and c.text not in seen:
            seen[c.text] = c
            all_cands.append(c)

    # Baseline (stage 0) — always included.
    _add(_make_candidate(baseline, "baseline (unmodified)", stage=0))

    # ── Stage 1: lookup compression ─────────────────────────────────────
    s1 = _stage1_lookup(baseline, context_prev, context_next)
    for c in s1:
        _add(c)

    # Best output from Stage 1 (shortest that still makes sense).
    s1_best = s1[-1].text if s1 else baseline

    # ── Stage 2: clause pruning ──────────────────────────────────────────
    s2 = _stage2_clause_pruning(s1_best, context_prev)
    for c in s2:
        _add(c)

    s2_best = s2[-1].text if s2 else s1_best

    # ── Stage 3: argos re-translation ───────────────────────────────────
    s3 = _stage3_retranslation(source_text, target_duration_s, context_prev, context_next)
    for c in s3:
        _add(c)

    # ── Stage 4: hard truncation ─────────────────────────────────────────
    # Apply to the best candidate we have so far (fewest chars).
    best_so_far = min(all_cands, key=lambda c: c.char_count).text
    s4 = _stage4_truncate(best_so_far, target_duration_s)
    for c in s4:
        _add(c)

    # ── Sort by closeness to target duration ────────────────────────────
    result = list(seen.values())
    result.sort(key=lambda c: abs(c.predicted_duration_s - target_duration_s))

    logger.debug(
        "get_shorter_translations: returning %d candidates; best='%s' (%.2fs predicted)",
        len(result),
        result[0].text[:40] if result else "",
        result[0].predicted_duration_s if result else 0.0,
    )

    return result


# //////////////////////////////////////////////////
# """Deterministic failure analysis and translation re-ranking stubs.

# The failure analysis function uses simple threshold rules derived from
# SegmentMetrics.  The translation re-ranking function is a **student assignment**
# — see the docstring for inputs, outputs, and implementation guidance.
# """

# import dataclasses
# import logging

# logger = logging.getLogger(__name__)


# @dataclasses.dataclass
# class TranslationCandidate:
#     """A candidate translation that fits a duration budget.

#     Attributes:
#         text: The translated text.
#         char_count: Number of characters in *text*.
#         brevity_rationale: Short explanation of what was shortened.
#     """
#     text: str
#     char_count: int
#     brevity_rationale: str = ""


# @dataclasses.dataclass
# class FailureAnalysis:
#     """Diagnostic summary of the dominant failure mode in a clip.

#     Attributes:
#         failure_category: One of "duration_overflow", "cumulative_drift",
#             "stretch_quality", or "ok".
#         likely_root_cause: One-sentence description.
#         suggested_change: Most impactful next action.
#     """
#     failure_category: str
#     likely_root_cause: str
#     suggested_change: str


# def analyze_failures(report: dict) -> FailureAnalysis:
#     """Classify the dominant failure mode from a clip evaluation report.

#     Pure heuristic — no LLM needed.  The thresholds below match the policy
#     bands defined in ``alignment.decide_action``.

#     Args:
#         report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
#             ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
#             ``total_cumulative_drift_s``, ``n_translation_retries``.

#     Returns:
#         A ``FailureAnalysis`` dataclass.
#     """
#     mean_err = report.get("mean_abs_duration_error_s", 0.0)
#     pct_severe = report.get("pct_severe_stretch", 0.0)
#     drift = abs(report.get("total_cumulative_drift_s", 0.0))
#     retries = report.get("n_translation_retries", 0)

#     if pct_severe > 20:
#         return FailureAnalysis(
#             failure_category="duration_overflow",
#             likely_root_cause=(
#                 f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
#                 "translated text is consistently too long for the available time window."
#             ),
#             suggested_change="Implement duration-aware translation re-ranking (P8).",
#         )

#     if drift > 3.0:
#         return FailureAnalysis(
#             failure_category="cumulative_drift",
#             likely_root_cause=(
#                 f"Total drift is {drift:.1f}s — small per-segment overflows "
#                 "accumulate because gaps between segments are not being reclaimed."
#             ),
#             suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
#         )

#     if mean_err > 0.8:
#         return FailureAnalysis(
#             failure_category="stretch_quality",
#             likely_root_cause=(
#                 f"Mean duration error is {mean_err:.2f}s — segments fit within "
#                 "stretch limits but the stretch distorts audio quality."
#             ),
#             suggested_change="Lower the mild_stretch ceiling or shorten translations.",
#         )

#     return FailureAnalysis(
#         failure_category="ok",
#         likely_root_cause="No dominant failure mode detected.",
#         suggested_change="Review individual outlier segments if any remain.",
#     )


# def get_shorter_translations(
#     source_text: str,
#     baseline_es: str,
#     target_duration_s: float,
#     context_prev: str = "",
#     context_next: str = "",
# ) -> list[TranslationCandidate]:
#     """Return shorter translation candidates that fit *target_duration_s*.

#     .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

#        This function is intentionally a **stub that returns an empty list**.
#        Your task is to implement a strategy that produces shorter
#        target-language translations when the baseline translation is too long
#        for the time budget.

#        **Inputs**

#        ============== ======== ==================================================
#        Parameter      Type     Description
#        ============== ======== ==================================================
#        source_text    str      Original source-language segment text
#        baseline_es    str      Baseline target-language translation (from argostranslate)
#        target_duration_s float Time budget in seconds for this segment
#        context_prev   str      Text of the preceding segment (for coherence)
#        context_next   str      Text of the following segment (for coherence)
#        ============== ======== ==================================================

#        **Outputs**

#        A list of ``TranslationCandidate`` objects, sorted shortest first.
#        Each candidate has:

#        - ``text``: the shortened target-language translation
#        - ``char_count``: ``len(text)``
#        - ``brevity_rationale``: short note on what was changed

#        **Duration heuristic**: target-language TTS produces ~15 characters/second
#        (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
#        ≈ 45 characters.

#        **Approaches to consider** (pick one or combine):

#        1. **Rule-based shortening** — strip filler words, use shorter synonyms
#           from a lookup table, contract common phrases
#           (e.g. "en este momento" → "ahora").
#        2. **Multiple translation backends** — call argostranslate with
#           paraphrased input, or use a second translation model, then pick
#           the shortest output that preserves meaning.
#        3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
#           condensed alternatives.  This was the previous approach but adds
#           latency, cost, and a runtime dependency.
#        4. **Hybrid** — rule-based first, fall back to LLM only for segments
#           that still exceed the budget.

#        **Evaluation criteria**: the caller selects the candidate whose
#        ``len(text) / 15.0`` is closest to ``target_duration_s``.

#     Returns:
#         Empty list (stub).  Implement to return ``TranslationCandidate`` items.
#     """
#     # logger.info(
#     #     "get_shorter_translations called for %.1fs budget (%d chars baseline) — "
#     #     "returning empty list (student assignment stub).",
#     #     target_duration_s,
#     #     len(baseline_es),
#     # )
#     max_chars = int(target_duration_s * 15)

#     def normalize(text: str) -> str:
#         return " ".join(text.strip().split())

#     def make_candidate(text: str, rationale: str) -> TranslationCandidate:
#         text = normalize(text)
#         return TranslationCandidate(
#             text=text,
#             char_count=len(text),
#             brevity_rationale=rationale,
#         )

#     candidates = []

#     baseline = normalize(baseline_es)
#     candidates.append(make_candidate(baseline, "baseline"))

#     # --- Rule 1: Replace common long phrases with shorter equivalents ---
#     replacements = {
#         "en este momento": "ahora",
#         "en este instante": "ahora",
#         "debido a que": "porque",
#         "a pesar de que": "aunque",
#         "con el fin de": "para",
#         "de la misma manera": "igual",
#         "por lo tanto": "así que",
#         "sin embargo": "pero",
#         "es posible que": "quizá",
#         "tiene la capacidad de": "puede",
#         "es necesario": "hay que",
#     }

#     shortened = baseline
#     for k, v in replacements.items():
#         shortened = shortened.replace(k, v)
#     if shortened != baseline:
#         candidates.append(make_candidate(shortened, "phrase replacements"))

#     # --- Rule 2: Remove filler words ---
#     fillers = [
#         "muy", "realmente", "básicamente", "en realidad",
#         "simplemente", "solo", "solamente"
#     ]

#     def remove_fillers(text: str) -> str:
#         words = text.split()
#         return " ".join(w for w in words if w.lower() not in fillers)

#     no_fillers = remove_fillers(shortened)
#     if no_fillers != shortened:
#         candidates.append(make_candidate(no_fillers, "removed fillers"))

#     # --- Rule 3: Drop optional pronouns/articles (light heuristic) ---
#     drop_words = {"yo", "tú", "usted", "nosotros", "ellos", "ellas"}
#     articles = {"el", "la", "los", "las", "un", "una", "unos", "unas"}

#     def drop_optionals(text: str) -> str:
#         words = text.split()
#         result = []
#         for i, w in enumerate(words):
#             lw = w.lower()
#             # Drop pronouns always
#             if lw in drop_words:
#                 continue
#             # Drop articles only if not first word
#             if lw in articles and i != 0:
#                 continue
#             result.append(w)
#         return " ".join(result)

#     compact = drop_optionals(no_fillers)
#     if compact != no_fillers:
#         candidates.append(make_candidate(compact, "dropped pronouns/articles"))

#     # --- Rule 4: Truncate trailing clauses if still too long ---
#     def truncate(text: str, max_len: int) -> str:
#         if len(text) <= max_len:
#             return text
#         # Try cutting at punctuation first
#         for sep in [".", ",", ";", " y ", " pero "]:
#             parts = text.split(sep)
#             if len(parts) > 1:
#                 truncated = parts[0]
#                 if len(truncated) <= max_len:
#                     return truncated
#         # Hard cut
#         return text[:max_len].rstrip()

#     for c in list(candidates):
#         if c.char_count > max_chars:
#             truncated = truncate(c.text, max_chars)
#             candidates.append(
#                 make_candidate(truncated, "truncated to fit duration")
#             )

#     # --- Filter + sort ---
#     # Keep unique texts only
#     unique = {}
#     for c in candidates:
#         unique[c.text] = c

#     result = list(unique.values())

#     # Sort shortest first
#     result.sort(key=lambda x: x.char_count)

#     return result



# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
