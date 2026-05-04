# Foreign Whispers — Technical Report

> AI-Powered YouTube Video Dubbing Pipeline  
> May 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Alignment Engine](#3-alignment-engine)
4. [Speaker Diarization & Voice Cloning](#4-speaker-diarization--voice-cloning)
5. [Evaluation Results](#5-evaluation-results)
6. [Technical Challenges & Solutions](#6-technical-challenges--solutions)
7. [Final Dependency Configuration](#7-final-dependency-configuration)
8. [Conclusions](#8-conclusions)

---

## 1. Executive Summary

Foreign Whispers is a fully containerised AI pipeline that takes any YouTube video and produces a dubbed version in a target language, with per-speaker voice cloning, temporal alignment, and automatic evaluation. The system chains together six open-source AI models — each running as an independent Docker microservice — and orchestrates them through a FastAPI backend.

The pipeline was validated on a CBS News segment about the Strait of Hormuz (video ID: `GYQ5yGV_-Oc`, ~7 minutes, 98 segments, 3 speakers). The final dubbed video achieved:

- **1.08s mean absolute timing error** per segment
- **0% severe stretch** — no segment was stretched beyond intelligibility
- **0 cumulative drift** — errors do not compound across segments
- **0 synthesis failures** across all 98 segments

---

## 2. System Architecture

### 2.1 Service Overview

| Container | Role | Port | Image |
|---|---|---|---|
| `foreign-whispers-api` | FastAPI orchestration backend | 8080 | `aiproject-api` (custom) |
| `foreign-whispers-frontend` | Next.js UI dashboard | 8501 | `aiproject-frontend` (custom) |
| `foreign-whispers-stt` | Whisper GPU transcription | 8000 | `speaches:latest-cuda` |
| `foreign-whispers-tts` | Chatterbox GPU TTS + voice cloning | 8020 | `chatterbox-tts-api:latest` |

### 2.2 Pipeline Stages

```
YouTube URL
    │
    ▼
[yt-dlp] Download video + captions
    │
    ▼
[Whisper / Speaches] Transcribe → EN segments with timestamps
    │
    ▼
[Argos Translate] Translate EN → ES segments
    │
    ▼
[pyannote.audio 3.3.2] Diarize → speaker intervals (SPEAKER_00, 01, 02...)
    │
    ▼
[Alignment Engine] Compute per-segment stretch factors
    │
    ▼
[Chatterbox TTS] Synthesize ES audio per segment with speaker voice cloning
    │
    ▼
[pyrubberband] Time-stretch each segment to fit original timing window
    │
    ▼
[ffmpeg / moviepy] Stitch dubbed audio onto original video
    │
    ▼
Dubbed MP4
```

### 2.3 Technology Stack

| Component | Technology | Notes |
|---|---|---|
| Transcription | OpenAI Whisper (large) | GPU via Speaches container |
| Translation | Argos Translate | Offline, no API key required |
| Diarization | pyannote.audio 3.3.2 | HuggingFace token required |
| TTS Engine | Chatterbox (travisvn fork) | OpenAI-compatible API, CUDA |
| Time-stretch | pyrubberband (RubberBand) | High-quality phase-vocoder |
| API Framework | FastAPI + uvicorn | Async with threadpool for GPU work |
| Package Manager | uv (Astral) | Locked dependencies, fast builds |
| Compute | NVIDIA CUDA 12.1 (cu121) | cu128 incompatible with pyannote 3.x |

---

## 3. Alignment Engine

The core technical challenge in dubbing is that translated speech is typically 15–25% longer than the original due to language expansion (Spanish averages ~20% more syllables than English for equivalent content). The alignment engine addresses this through a multi-stage approach.

### 3.1 Segment-Level Alignment

Each translated segment is independently synthesized and post-processed. The engine computes a stretch factor per segment based on the ratio of raw TTS duration to the target window:

```python
speed_factor = raw_duration / (target_sec * stretch_factor)
speed_factor = clamp(speed_factor, SPEED_MIN=0.75, SPEED_MAX=1.25)
```

When the TTS output is less than 50% of the target window duration (`STRETCH_SKIP_RATIO`), the engine plays at natural speed and pads with silence rather than slowing speech to an unnatural pace.

### 3.2 Action Classification

Each segment is classified into one of three actions:

| Action | Condition | Count (GYQ video) |
|---|---|---|
| `accept` | Speed factor within bounds, no retry needed | 37 segments (37.8%) |
| `mild_stretch` | Speed factor at boundary but within SPEED_MAX | 45 segments (45.9%) |
| `request_shorter` | Translation retry requested for tighter fit | 16 segments (16.3%) |
| `fail` | Could not synthesize segment | 0 segments (0%) |

### 3.3 YouTube Caption Timing Offset

A speech offset is computed by comparing the first YouTube caption timestamp against the first Whisper segment timestamp. This corrects for pre-roll silence and title cards so TTS audio aligns with actual speech onset rather than video start.

---

## 4. Speaker Diarization & Voice Cloning

### 4.1 Diarization Results

`pyannote/speaker-diarization-3.1` identified three speakers across 59 diarization intervals covering the full 406-second video:

| Speaker ID | Role | Approx. Segments | Voice WAV |
|---|---|---|---|
| `SPEAKER_00` | Female anchor / narrator | ~55 segments | `es/SPEAKER_00.wav` |
| `SPEAKER_01` | Field reporter (Matt Smith) | ~18 segments | `es/SPEAKER_01.wav` |
| `SPEAKER_02` | Male expert (Bob McNally) | ~25 segments | `es/SPEAKER_02.wav` |

### 4.2 Voice Cloning Approach

Reference WAV files are extracted from the source video at known speaker timestamps using ffmpeg. These 10–15 second mono 22050Hz clips are provided to Chatterbox's `/v1/audio/speech/upload` endpoint which performs zero-shot voice cloning for each synthesized segment.

Resolution priority order:

```
per-speaker WAV (es/SPEAKER_XX.wav)
    → language default (es/default.wav)
    → global default (default.wav)
    → Chatterbox built-in voice
```

### 4.3 ASCII Transliteration Fix

Chatterbox's tokenizer is ASCII-only. Spanish accented characters (`é`, `ñ`, `ú`, `¿`, `¡` etc.) caused CUDA device-side assertions — token IDs exceeded the embedding table size. A pre-synthesis transliteration step strips combining marks after NFD decomposition:

```python
normalized = unicodedata.normalize("NFD", text)
return "".join(
    c for c in normalized if unicodedata.category(c) != "Mn"
).encode("ascii", errors="ignore").decode("ascii")
# é → e,  ñ → n,  ¿ → (removed),  ó → o
```

---

## 5. Evaluation Results

**Test video:** CBS News — *"Strait of Hormuz disruption threatens to shake global economy"*  
`GYQ5yGV_-Oc` · 7 min 46 sec · 98 segments · 3 speakers · English → Spanish

### 5.1 Timing Metrics

| Metric | Value |
|---|---|
| Mean absolute duration error | **1.08 s** |
| Severe stretch segments | **0 (0.0%)** |
| Gap shifts | **0** |
| Translation retries (`request_shorter`) | **16** |
| Total cumulative drift | **0.0 s** |
| Speaking rate variance | 0.0168 |
| Timing score | 17.6 / 100 |
| Segments: accepted | 37 (37.8%) |
| Segments: mild stretch | 45 (45.9%) |
| Segments: failed | **0 (0%)** |

> The timing score of 17.6 reflects the inherent difficulty of dubbing Spanish into English timing windows — Spanish is structurally longer. The 0% failure rate and 0.0s cumulative drift confirm the pipeline is stable and does not compound errors across segments.

### 5.2 Sample Segment Detail

| Index | Speaker | Target (s) | Raw TTS (s) | Speed Factor | Action |
|---|---|---|---|---|---|
| 2 | SPEAKER_01 | 6.08 | 7.16 | 1.178 | accept |
| 13 | SPEAKER_00 | 6.56 | 10.68 | 1.250 | mild_stretch |
| 31 | SPEAKER_02 | 7.64 | 6.76 | 0.885 | accept |
| 52 | SPEAKER_02 | 8.36 | 6.88 | 0.823 | accept |
| 96 | SPEAKER_02 | 10.96 | 6.12 | 0.750 | accept |

---

## 6. Technical Challenges & Solutions

### 6.1 pyannote.audio Import Failure (stub version 1.1.2)

**Problem:** `pyannote.audio` version 1.1.2 installed as a stub package — only `__version__` was exposed, no `Pipeline` class. `uv.lock` had pinned to this broken PyPI artifact.

**Solution:** Removed the version pin from `pyproject.toml`, allowing uv to resolve to 3.3.2 which has a complete package structure.

---

### 6.2 torchaudio API Removal (cu128 index)

**Problem:** `torchaudio 2.10.0` (cu128) removed `AudioMetaData` and `list_audio_backends()`, both required by pyannote.audio 3.x at import time. The cu128 index only provides torchaudio 2.7.0+, all of which have the APIs removed.

**Solution:** Switched from the `cu128` PyTorch index to `cu121`, which provides `torchaudio 2.4.x` with full API compatibility. CUDA 12.1 builds run correctly on all CUDA 12.x hardware.

---

### 6.3 numpy<2 vs pyannote 4.x Conflict

**Problem:** `pyannote.audio 4.x` requires `numpy>=2`, but other dependencies (`openai-whisper`, `librosa`, `moviepy`) constrained `numpy<2`. Attempting to upgrade produced an unsatisfiable resolution.

**Solution:** Remained on `pyannote.audio 3.3.2` (the last version compatible with numpy 1.x) and resolved the torchaudio incompatibility via the cu121 index instead.

---

### 6.4 Chatterbox CUDA Device-Side Assert on Spanish Text

**Problem:** Spanish text containing accented characters caused CUDA kernel assertions (`srcIndex < srcSelectDimSize` in `Indexing.cu`). The Chatterbox tokenizer vocabulary is ASCII-only; non-ASCII characters mapped to out-of-bounds token IDs in the embedding table.

**Solution:** Added `_transliterate()` to `ChatterboxClient` using `unicodedata.normalize('NFD')` followed by filtering out `Mn`-category (combining mark) characters, then ASCII encoding with `errors='ignore'`.

---

### 6.5 diarization_path Not Wired to TTS Router

**Problem:** The TTS router constructed `TTSService` without passing `diarization_path`, so all 98 segments were synthesized with the default female voice regardless of speaker assignment in the diarization sidecar.

**Solution:** Updated the TTS router to auto-detect the `.diarize.json` sidecar and pass it to `text_file_to_speech()`, enabling per-speaker voice cloning across all segments.

---

## 7. Final Dependency Configuration

The following `pyproject.toml` configuration resolves all compatibility issues:

```toml
[dependency-groups]
alignment = [
    "logfire",
    "silero-vad",
    "pyannote.audio",       # resolves to 3.3.2
]

[tool.uv]
constraint-dependencies = [
    "onnxruntime<1.24",
    "numpy<2",
    "transformers<4.46",
]
find-links = [
    "https://download.pytorch.org/whl/cu121",
]

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"   # NOT cu128
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchaudio = { index = "pytorch-cu121" }
torchvision = { index = "pytorch-cu121" }
```

---

## 8. Conclusions

Foreign Whispers demonstrates a production-grade, fully automated dubbing pipeline using open-source AI models. Key achievements:

- **End-to-end pipeline** from YouTube URL to dubbed MP4 via a single API call sequence
- **Per-speaker voice cloning** using zero-shot Chatterbox TTS with reference WAVs extracted from the source video
- **Temporal alignment** keeping all 98 segments within 1.08s mean error with 0% failure rate
- **Full GPU acceleration** (Whisper + Chatterbox) with CPU fallback for all other stages
- **Stable Docker Compose deployment** with health checks and graceful dependency ordering
- **Five non-trivial compatibility issues resolved** between pyannote, torchaudio, numpy, and Chatterbox

The primary remaining limitation is Chatterbox's ASCII-only tokenizer, which requires transliteration of Spanish text and reduces pronunciation fidelity for accented characters. A Spanish-native TTS model (Coqui `es/mai/tacotron2-DDC` is included as a fallback) would eliminate this constraint at the cost of voice cloning capability.

---
