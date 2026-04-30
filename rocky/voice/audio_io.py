from __future__ import annotations

from collections import deque
import tempfile
import wave
from pathlib import Path

from rocky.voice.config import VoiceConfig
from rocky.voice.stt import VoiceDependencyError


class FixedWindowRecorder:
    def __init__(self, config: VoiceConfig):
        self.config = config

    def record_to_file(self) -> Path:
        try:
            import numpy as np
            import sounddevice as sd
        except Exception as error:  # pragma: no cover - import guard
            raise VoiceDependencyError("Missing microphone dependencies. Install sounddevice and numpy.") from error

        if self.config.record_mode == "vad":
            return self._record_until_silence(np, sd)
        return self._record_fixed_window(np, sd)

    def _record_fixed_window(self, np, sd) -> Path:
        frame_count = int(self.config.sample_rate * self.config.record_seconds)
        recording = sd.rec(
            frame_count,
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return self._write_wav(recording[:, 0], np)

    def _record_until_silence(self, np, sd) -> Path:
        block_duration = 0.1
        block_size = max(1, int(self.config.sample_rate * block_duration))
        max_blocks = max(1, int(self.config.record_seconds / block_duration))
        min_blocks = max(1, int(self.config.min_record_seconds / block_duration))
        silence_blocks_needed = max(1, int(self.config.silence_seconds / block_duration))
        preroll_blocks = max(0, int(self.config.preroll_seconds / block_duration))
        preroll = deque(maxlen=preroll_blocks)
        silence_blocks = 0
        speech_blocks = 0
        heard_speech = False
        chunks = []

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
        ) as stream:
            for _ in range(max_blocks):
                block, _ = stream.read(block_size)
                samples = block[:, 0].copy()
                rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0

                if rms >= self.config.silence_threshold:
                    if not heard_speech:
                        chunks.extend(preroll)
                        preroll.clear()
                    heard_speech = True
                    silence_blocks = 0
                elif not heard_speech:
                    preroll.append(samples)
                    continue
                else:
                    silence_blocks += 1

                chunks.append(samples)
                speech_blocks += 1
                if speech_blocks >= min_blocks and silence_blocks >= silence_blocks_needed:
                    break

        if not chunks:
            return self._write_wav(np.zeros(block_size, dtype="float32"), np)
        return self._write_wav(np.concatenate(chunks), np)

    def _write_wav(self, samples, np) -> Path:
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16)

        handle = tempfile.NamedTemporaryFile(prefix="rocky-user-", suffix=".wav", delete=False)
        path = Path(handle.name)
        handle.close()

        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.config.sample_rate)
            wav.writeframes(pcm.tobytes())
        return path


class AudioPlayer:
    def play_file(self, audio_path: str | Path) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf
        except Exception as error:  # pragma: no cover - import guard
            raise VoiceDependencyError("Missing playback dependencies. Install sounddevice and soundfile.") from error

        data, sample_rate = sf.read(str(audio_path), dtype="float32")
        sd.play(data, sample_rate)
        sd.wait()
