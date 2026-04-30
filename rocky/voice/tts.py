from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Protocol

from rocky.voice.config import VoiceConfig
from rocky.voice.stt import VoiceDependencyError


class TextToSpeech(Protocol):
    def synthesize_to_file(self, text: str, output_path: str | Path) -> Path:
        raise NotImplementedError


class KokoroTTS:
    def __init__(self, lang_code: str, voice: str, speed: float = 1.0):
        try:
            from kokoro import KPipeline
        except Exception as error:  # pragma: no cover - import guard
            raise VoiceDependencyError(
                "Missing TTS dependency. Install kokoro and soundfile, or set ROCKY_VOICE_TTS=piper."
            ) from error

        self.voice = voice
        self.speed = speed
        self.pipeline = KPipeline(lang_code=lang_code)

    def synthesize_to_file(self, text: str, output_path: str | Path) -> Path:
        try:
            import numpy as np
            import soundfile as sf
        except Exception as error:  # pragma: no cover - import guard
            raise VoiceDependencyError("Missing TTS audio dependencies. Install numpy and soundfile.") from error

        chunks = []
        for _, _, audio in self.pipeline(text, voice=self.voice, speed=self.speed, split_pattern=r"\n+"):
            chunks.append(audio)
        if not chunks:
            raise RuntimeError("TTS produced no audio.")

        path = Path(output_path)
        audio = np.concatenate(chunks)
        sf.write(path, audio, 24000)
        return path


class PiperTTS:
    def __init__(self, model_path: str, executable: str = "piper"):
        if not model_path:
            raise VoiceDependencyError("Set ROCKY_PIPER_MODEL to a Piper voice model path.")
        model = Path(model_path)
        if not model.exists():
            raise VoiceDependencyError(f"Piper voice model not found: {model}")
        config = model.with_suffix(model.suffix + ".json")
        if not config.exists():
            raise VoiceDependencyError(
                f"Piper voice config not found: {config}. Download the matching .onnx.json file too."
            )
        self.model_path = str(model)
        self.executable = executable

    def synthesize_to_file(self, text: str, output_path: str | Path) -> Path:
        path = Path(output_path)
        command = [
            self.executable,
            "--model",
            self.model_path,
            "--output_file",
            str(path),
        ]
        try:
            subprocess.run(command, input=text, text=True, check=True, capture_output=True)
        except FileNotFoundError as error:
            raise VoiceDependencyError(
                "Missing Piper executable. Install it with: poetry run pip install piper-tts"
            ) from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "").strip()
            raise RuntimeError(f"Piper failed: {detail}") from error
        return path


class MacSayTTS:
    def __init__(self, voice: str = "", speed: float = 1.0):
        if sys.platform != "darwin":
            raise VoiceDependencyError("The say TTS backend is only available on macOS.")
        self.voice = voice
        self.rate = max(80, min(420, int(180 * speed)))

    def synthesize_to_file(self, text: str, output_path: str | Path) -> Path:
        path = Path(output_path)
        command = ["say", "--data-format=LEI16@22050", "-r", str(self.rate), "-o", str(path)]
        if self.voice:
            command.extend(["-v", self.voice])
        command.append(text)
        try:
            subprocess.run(command, check=True, capture_output=True)
        except FileNotFoundError as error:
            raise VoiceDependencyError("Missing macOS say command. Choose another TTS backend.") from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or b"").decode(errors="replace").strip()
            raise RuntimeError(f"say failed: {detail}") from error
        return path


class NullTTS:
    def synthesize_to_file(self, text: str, output_path: str | Path) -> Path:
        raise VoiceDependencyError("No usable TTS backend is configured.")


def build_tts(config: VoiceConfig) -> TextToSpeech:
    if config.tts_backend == "say":
        return MacSayTTS(voice=config.tts_voice, speed=config.tts_speed)
    if config.tts_backend == "kokoro":
        return KokoroTTS(
            lang_code=config.tts_language,
            voice=config.tts_voice,
            speed=config.tts_speed,
        )
    if config.tts_backend == "piper":
        return PiperTTS(config.piper_model_path, executable=config.piper_executable)
    if config.tts_backend in {"off", "none"}:
        return NullTTS()
    raise VoiceDependencyError(f"Unsupported TTS backend: {config.tts_backend}")
