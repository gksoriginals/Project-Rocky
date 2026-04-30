from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rocky.voice.config import VoiceConfig


class VoiceDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str | None = None


class SpeechToText(Protocol):
    def transcribe(self, audio_path: str | Path) -> Transcript:
        raise NotImplementedError


class FasterWhisperSTT:
    def __init__(self, model_name: str, language: str | None = None):
        try:
            from faster_whisper import WhisperModel
        except Exception as error:  # pragma: no cover - import guard
            raise VoiceDependencyError(
                "Missing STT dependency. Install faster-whisper or choose another voice STT backend."
            ) from error

        self.language = language
        self.model = WhisperModel(model_name, device="auto", compute_type="auto")

    def transcribe(self, audio_path: str | Path) -> Transcript:
        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=True,
            beam_size=5,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        language = getattr(info, "language", None)
        return Transcript(text=text, language=str(language) if language else None)


class NullSTT:
    def transcribe(self, audio_path: str | Path) -> Transcript:
        raise VoiceDependencyError(
            "No usable STT backend is configured. Set ROCKY_VOICE_STT=faster-whisper and install faster-whisper."
        )


def build_stt(config: VoiceConfig) -> SpeechToText:
    if config.stt_backend in {"faster-whisper", "whisper", "whisper.cpp"}:
        return FasterWhisperSTT(config.stt_model, language=config.stt_language)
    if config.stt_backend in {"off", "none"}:
        return NullSTT()
    raise VoiceDependencyError(f"Unsupported STT backend: {config.stt_backend}")
