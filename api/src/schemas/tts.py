"""Pydantic schemas for the TTS API contract."""

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """Optional request body for the TTS endpoint.

    All fields are optional — the endpoint works with no body (query-param
    only) for backwards compatibility.

    Attributes:
        use_diarization: When ``True``, the service will look for a
            ``.diarize.json`` sidecar next to the translated transcript and
            synthesize each segment with the reference voice of the dominant
            speaker.  Requires that ``speakers_dir`` is configured on the
            server and that diarization has been run for this video.
        target_language: BCP-47 language code used for voice directory lookup
            (default ``"es"``).  Must match the language of the translated
            transcript.
    """

    use_diarization: bool = Field(
        default=False,
        description=(
            "Enable per-speaker voice cloning. "
            "Requires a .diarize.json sidecar and a configured speakers_dir."
        ),
    )
    target_language: str = Field(
        default="es",
        description="BCP-47 language code for voice directory lookup.",
    )


class TTSSegmentVoice(BaseModel):
    """Per-segment voice assignment included in the response when diarization is active.

    Attributes:
        index:       Zero-based segment index.
        speaker_id:  Diarization label (e.g. ``"SPEAKER_00"``), or ``None``
                     when the segment could not be assigned to any speaker.
        speaker_wav: Relative path to the reference WAV that was used, or
                     ``""`` when the server's built-in default voice was used.
    """

    index: int
    speaker_id: str | None
    speaker_wav: str


class TTSResponse(BaseModel):
    """Response from the TTS synthesis endpoint.

    Attributes:
        video_id:       The video identifier.
        audio_path:     Path to the generated WAV file (relative to ui_dir).
        speakers_used:  List of unique speaker IDs that were synthesized with
                        a per-speaker reference voice.  Empty list when
                        diarization was not used or no reference WAVs were found.
        segment_voices: Per-segment voice assignment detail.  Only populated
                        when ``use_diarization=True`` was requested; otherwise
                        an empty list.
    """

    video_id: str
    audio_path: str
    speakers_used: list[str] = Field(
        default_factory=list,
        description="Unique speaker IDs that used a reference voice WAV.",
    )
    segment_voices: list[TTSSegmentVoice] = Field(
        default_factory=list,
        description="Per-segment speaker/voice assignments (populated when diarization is active).",
    )