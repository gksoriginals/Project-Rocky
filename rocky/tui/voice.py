from __future__ import annotations

import queue
import re
import tempfile
import threading
import time
from pathlib import Path


class TUIVoiceManager:
    def __init__(self, tui):
        self.tui = tui
        self.active = False
        self.input_queue = queue.Queue()
        self.sentence_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.turn_completed_event = threading.Event()
        self.turn_completed_event.set()

        self.stt = None
        self.tts = None
        self.recorder = None
        self.player = None
        self.config = None

        self.voice_thread = None
        self.tts_thread = None

        self.spoken_cursor = 0
        self.current_text = ""
        self.is_tool_call = False
        self._tts_busy = False

    def toggle(self) -> None:
        if self.active:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        if self.active:
            return
        
        try:
            self._load_backends()
        except Exception as e:
            self.tui.session_state.set_notice(f"Voice dependency error: {e}")
            self.tui.session_state.add_trace_entry(
                phase="voice", summary="Init Error", detail=str(e), turn_index=self.tui.session_state.turn_index
            )
            return
            
        self.active = True
        self.stop_event.clear()
        self.turn_completed_event.set()

        self.voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
        self.tts_thread = threading.Thread(target=self._tts_loop, daemon=True)

        self.tts_thread.start()
        self.voice_thread.start()
        self.tui.session_state.set_notice("Continuous voice mode started.")

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        self.stop_event.set()
        self.sentence_queue.put(None)
        self.turn_completed_event.set()
        if self.player:
            self.player.stop()
        self.tui.session_state.set_notice("Continuous voice mode stopped.")

    def _load_backends(self) -> None:
        from rocky.voice.config import VoiceConfig
        from rocky.voice.stt import build_stt
        from rocky.voice.tts import build_tts
        from rocky.voice.audio_io import FixedWindowRecorder, AudioPlayer
        
        if self.config is None:
            self.config = VoiceConfig.from_env()
        if self.stt is None:
            self.stt = build_stt(self.config)
        if self.tts is None:
            self.tts = build_tts(self.config)
        if self.recorder is None:
            self.recorder = FixedWindowRecorder(self.config)
        if self.player is None:
            self.player = AudioPlayer()

    def _voice_loop(self) -> None:
        while not self.stop_event.is_set():
            # Wait for any active turn and TTS to finish
            self.turn_completed_event.wait()
            if self.stop_event.is_set():
                break

            self.tui.session_state.set_notice("Listening...")
            try:
                self.tui.session_state.add_trace_entry(
                    phase="voice", summary="VAD", detail="Waiting for speech...", turn_index=self.tui.session_state.turn_index
                )
                audio_path = self.recorder.record_to_file()
            except Exception as e:
                if not self.stop_event.is_set():
                    self.tui.session_state.add_trace_entry(
                        phase="voice", summary="Recording Error", detail=str(e), turn_index=self.tui.session_state.turn_index
                    )
                time.sleep(1)
                continue

            if self.stop_event.is_set():
                Path(audio_path).unlink(missing_ok=True)
                break

            self.tui.session_state.add_trace_entry(
                phase="voice", summary="STT", detail="Transcribing audio...", turn_index=self.tui.session_state.turn_index
            )
            try:
                assert self.stt is not None
                transcript = self.stt.transcribe(audio_path)
            except Exception as e:
                self.tui.session_state.add_trace_entry(
                    phase="voice", summary="STT Error", detail=str(e), turn_index=self.tui.session_state.turn_index
                )
                continue
            finally:
                if not self.config.keep_audio:
                    Path(audio_path).unlink(missing_ok=True)

            text = transcript.text.strip()
            self.tui.session_state.add_trace_entry(
                phase="voice", summary="STT Result", detail=text or "[Silence]", turn_index=self.tui.session_state.turn_index
            )
            if not text:
                continue

            self.turn_completed_event.clear()
            self.input_queue.put(text)

    def _tts_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                sentence = self.sentence_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            
            if sentence is None:
                continue

            if sentence == "<TURN_COMPLETED>":
                self.turn_completed_event.set()
                continue
            
            if sentence.strip():
                self._tts_busy = True
                output = tempfile.NamedTemporaryFile(prefix="rocky-reply-", suffix=".wav", delete=False)
                output_path = Path(output.name)
                output.close()
                try:
                    assert self.tts is not None
                    self.tts.synthesize_to_file(sentence, output_path)
                    assert self.player is not None
                    self.player.play_file(output_path)
                except Exception as e:
                    pass
                finally:
                    if not self.config.keep_audio:
                        output_path.unlink(missing_ok=True)
                self._tts_busy = False

    def process_delta(self, new_text: str) -> None:
        if not self.active:
            return

        if len(new_text) < len(self.current_text):
            if not self.is_tool_call and self.current_text[self.spoken_cursor:].strip():
                self.sentence_queue.put(self.current_text[self.spoken_cursor:].strip())
            self.spoken_cursor = 0
            self.is_tool_call = False

        self.current_text = new_text

        if "<tool" in self.current_text or "<thought" in self.current_text:
            self.is_tool_call = True
            return

        if self.is_tool_call:
            return

        unspoken = self.current_text[self.spoken_cursor:]
        match = re.search(r'(?<=[.!?])\s+', unspoken)
        while match:
            sentence_end = match.start()
            sentence = unspoken[:sentence_end].strip()
            if sentence:
                self.sentence_queue.put(sentence)
            self.spoken_cursor += sentence_end + len(match.group())
            unspoken = self.current_text[self.spoken_cursor:]
            match = re.search(r'(?<=[.!?])\s+', unspoken)

    def process_turn_end(self) -> None:
        if not self.active:
            return
            
        if not self.is_tool_call:
            remaining = self.current_text[self.spoken_cursor:].strip()
            if remaining:
                self.sentence_queue.put(remaining)
                
        self.spoken_cursor = 0
        self.current_text = ""
        self.is_tool_call = False
        
        # Signal that the turn is over, so TTS worker can unblock recording when queue is empty
        self.sentence_queue.put("<TURN_COMPLETED>")
