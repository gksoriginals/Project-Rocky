from __future__ import annotations

import os
import sys
from dataclasses import dataclass


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    stt_backend: str = "faster-whisper"
    stt_model: str = "large-v3-turbo"
    stt_language: str | None = None
    tts_backend: str = "say" if sys.platform == "darwin" else "piper"
    tts_voice: str = "af_heart"
    tts_language: str = "a"
    tts_speed: float = 1.0
    piper_model_path: str = ""
    piper_executable: str = "piper"
    sample_rate: int = 16000
    listen_mode: str = "push_to_talk"
    record_mode: str = "vad"
    record_seconds: float = 12.0
    min_record_seconds: float = 1.5
    silence_seconds: float = 1.6
    silence_threshold: float = 0.008
    preroll_seconds: float = 0.3
    keep_audio: bool = False

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        language = os.getenv("ROCKY_VOICE_STT_LANGUAGE") or None
        return cls(
            stt_backend=os.getenv("ROCKY_VOICE_STT", "faster-whisper").strip().lower(),
            stt_model=os.getenv("ROCKY_VOICE_STT_MODEL", "large-v3-turbo").strip(),
            stt_language=language.strip() if language else None,
            tts_backend=os.getenv("ROCKY_VOICE_TTS", cls.tts_backend).strip().lower(),
            tts_voice=os.getenv("ROCKY_VOICE_NAME", "af_heart").strip(),
            tts_language=os.getenv("ROCKY_VOICE_TTS_LANGUAGE", "a").strip(),
            tts_speed=_float_env("ROCKY_VOICE_TTS_SPEED", 1.0),
            piper_model_path=os.getenv("ROCKY_PIPER_MODEL", "").strip(),
            piper_executable=os.getenv("ROCKY_PIPER_EXECUTABLE", "piper").strip() or "piper",
            sample_rate=_int_env("ROCKY_VOICE_SAMPLE_RATE", 16000),
            listen_mode=os.getenv("ROCKY_VOICE_LISTEN_MODE", "push_to_talk").strip().lower(),
            record_mode=os.getenv("ROCKY_VOICE_RECORD_MODE", "vad").strip().lower(),
            record_seconds=_float_env("ROCKY_VOICE_RECORD_SECONDS", 12.0),
            min_record_seconds=_float_env("ROCKY_VOICE_MIN_RECORD_SECONDS", 1.5),
            silence_seconds=_float_env("ROCKY_VOICE_SILENCE_SECONDS", 1.6),
            silence_threshold=_float_env("ROCKY_VOICE_SILENCE_THRESHOLD", 0.008),
            preroll_seconds=_float_env("ROCKY_VOICE_PREROLL_SECONDS", 0.3),
            keep_audio=os.getenv("ROCKY_VOICE_KEEP_AUDIO", "").strip().lower() in {"1", "true", "yes"},
        )
