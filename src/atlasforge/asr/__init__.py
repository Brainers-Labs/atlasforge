"""ASR helpers: audio decoding, long-audio chunking and merging."""

from atlasforge.asr.audio import decode_audio, duration_s, pcm_to_float32, pcm_to_wav
from atlasforge.asr.chunking import (
    frame_levels,
    merge_texts,
    plan_windows,
    plan_windows_silence_aware,
    transcribe_long,
)
from atlasforge.asr.models import ASR_MODELS, accept_licence_url, asr_model_for

__all__ = [
    "ASR_MODELS",
    "accept_licence_url",
    "asr_model_for",
    "decode_audio",
    "duration_s",
    "frame_levels",
    "merge_texts",
    "pcm_to_float32",
    "pcm_to_wav",
    "plan_windows",
    "plan_windows_silence_aware",
    "transcribe_long",
]
