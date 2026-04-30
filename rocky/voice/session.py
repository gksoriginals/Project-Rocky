from __future__ import annotations

import tempfile
import time
from pathlib import Path

from rocky.agent import RockyAgent
from rocky.voice.audio_io import AudioPlayer, FixedWindowRecorder
from rocky.voice.config import VoiceConfig
from rocky.voice.stt import SpeechToText, VoiceDependencyError, build_stt
from rocky.voice.tts import TextToSpeech, build_tts


class VoiceSession:
    def __init__(
        self,
        agent: RockyAgent,
        config: VoiceConfig | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        recorder: FixedWindowRecorder | None = None,
        player: AudioPlayer | None = None,
    ):
        self.agent = agent
        self.config = config or VoiceConfig.from_env()
        self.stt = stt
        self.tts = tts
        self.recorder = recorder or FixedWindowRecorder(self.config)
        self.player = player or AudioPlayer()

    def load_backends(self) -> None:
        if self.stt is None:
            print(f"Loading STT: {self.config.stt_backend} ({self.config.stt_model})...", flush=True)
            self.stt = build_stt(self.config)
        if self.tts is None:
            print(f"Loading TTS: {self.config.tts_backend}...", flush=True)
            self.tts = build_tts(self.config)

    def run_once(self) -> str:
        self.load_backends()
        audio_path = self.recorder.record_to_file()
        try:
            assert self.stt is not None
            transcript = self.stt.transcribe(audio_path)
        finally:
            if not self.config.keep_audio:
                Path(audio_path).unlink(missing_ok=True)

        user_text = transcript.text.strip()
        if not user_text:
            return ""

        import queue
        import threading
        import re

        sentence_queue = queue.Queue()
        stop_event = threading.Event()

        def tts_worker():
            try:
                while not stop_event.is_set():
                    try:
                        sentence = sentence_queue.get(timeout=0.1)
                        if sentence is None:
                            break
                        if sentence.strip():
                            # Print the sentence as it is being spoken
                            print(f"Rocky: {sentence}", flush=True)
                            output = tempfile.NamedTemporaryFile(prefix="rocky-reply-", suffix=".wav", delete=False)
                            output_path = Path(output.name)
                            output.close()
                            try:
                                assert self.tts is not None
                                self.tts.synthesize_to_file(sentence, output_path)
                                self.player.play_file(output_path)
                            finally:
                                if not self.config.keep_audio:
                                    output_path.unlink(missing_ok=True)
                    except queue.Empty:
                        continue
            except Exception as e:
                print(f"\n[TTS Error] {e}")

        tts_thread = threading.Thread(target=tts_worker, daemon=True)
        tts_thread.start()

        class StreamState:
            spoken_cursor = 0
            current_text = ""
            is_tool_call = False

        state = StreamState()

        def on_event(event):
            if event.type == "assistant_delta":
                new_text = str(event.payload.get("content") or "")
                
                # Detect new turn
                if len(new_text) < len(state.current_text):
                    if not state.is_tool_call and state.current_text[state.spoken_cursor:].strip():
                        sentence_queue.put(state.current_text[state.spoken_cursor:].strip())
                    state.spoken_cursor = 0
                    state.is_tool_call = False
                
                state.current_text = new_text
                
                if "<tool" in state.current_text or "<thought" in state.current_text:
                    state.is_tool_call = True
                    return
                
                if state.is_tool_call:
                    return
                
                unspoken = state.current_text[state.spoken_cursor:]
                match = re.search(r'(?<=[.!?])\s+', unspoken)
                while match:
                    sentence_end = match.start()
                    sentence = unspoken[:sentence_end].strip()
                    if sentence:
                        sentence_queue.put(sentence)
                    state.spoken_cursor += sentence_end + len(match.group())
                    unspoken = state.current_text[state.spoken_cursor:]
                    match = re.search(r'(?<=[.!?])\s+', unspoken)

        for _ in self.agent.process_turn(user_text, on_event=on_event):
            pass

        # Flush any remaining text from the final turn
        if not state.is_tool_call:
            remaining = state.current_text[state.spoken_cursor:].strip()
            if remaining:
                sentence_queue.put(remaining)

        sentence_queue.put(None)
        tts_thread.join()

        return self.agent.session_state.last_answer.strip()

    def run(self) -> int:
        print("Rocky voice mode")
        print(f"STT: {self.config.stt_backend} ({self.config.stt_model})")
        print(f"TTS: {self.config.tts_backend} ({self.config.tts_voice})")
        print(f"Listen mode: {self.config.listen_mode}")
        print(f"Recording: {self.config.record_mode}, max {self.config.record_seconds:g}s")
        try:
            self.load_backends()
        except VoiceDependencyError as error:
            print(str(error))
            return 1
        if self.config.listen_mode in {"continuous", "hands_free", "hands-free"}:
            return self._run_continuous()
        return self._run_push_to_talk()

    def _run_push_to_talk(self) -> int:
        print("Press Enter to record, or type q then Enter to quit.")

        while True:
            command = input("\nvoice> ").strip().lower()
            if command in {"q", "quit", "exit"}:
                return 0
            try:
                if self.config.record_mode == "vad":
                    print("Recording... speak now; stopping after silence.")
                else:
                    print(f"Recording for {self.config.record_seconds:g}s...")
                self.run_once()
            except VoiceDependencyError as error:
                print(str(error))
                return 1

    def _run_continuous(self) -> int:
        print("Continuous listening. Press Ctrl-C to quit.")
        while True:
            try:
                if self.config.record_mode == "vad":
                    print("\nListening... speak when ready.")
                else:
                    print(f"\nRecording for {self.config.record_seconds:g}s...")
                self.run_once()
                time.sleep(0.15)
            except KeyboardInterrupt:
                print("\nVoice mode stopped.")
                return 0
            except VoiceDependencyError as error:
                print(str(error))
                return 1
