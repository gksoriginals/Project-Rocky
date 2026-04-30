import unittest
from unittest.mock import Mock, MagicMock
from rocky.tui.voice import TUIVoiceManager
from rocky.voice.config import VoiceConfig

class FakeSessionState:
    def __init__(self):
        self.turn_index = 1
        self.notice = ""
        self.traces = []

    def set_notice(self, text):
        self.notice = text

    def add_trace_entry(self, phase, summary, detail, turn_index):
        self.traces.append({"phase": phase, "summary": summary, "detail": detail, "turn_index": turn_index})

class FakeTUI:
    def __init__(self):
        self.session_state = FakeSessionState()

class TestTUIVoiceManager(unittest.TestCase):
    def setUp(self):
        self.tui = FakeTUI()
        self.manager = TUIVoiceManager(self.tui)

    def test_voice_manager_init(self):
        self.assertFalse(self.manager.active)
        self.assertEqual(self.manager.current_text, "")
        self.assertTrue(self.manager.input_queue.empty())
        self.assertTrue(self.manager.sentence_queue.empty())

    def test_process_delta_chunks_sentences(self):
        self.manager.active = True
        
        self.manager.process_delta("Hello. ")
        self.assertEqual(self.manager.sentence_queue.get(timeout=1), "Hello.")
        
        self.manager.process_delta("Hello. How are ")
        self.assertTrue(self.manager.sentence_queue.empty())
        
        self.manager.process_delta("Hello. How are you? ")
        self.assertEqual(self.manager.sentence_queue.get(timeout=1), "How are you?")

    def test_process_delta_ignores_tool_calls(self):
        self.manager.active = True
        self.manager.process_delta("Hello. ")
        
        self.assertEqual(self.manager.sentence_queue.get(timeout=1), "Hello.")
        
        self.manager.process_delta("Hello. <tool_call>analyze()</tool_call> ")
        self.assertTrue(self.manager.sentence_queue.empty())
        self.assertTrue(self.manager.is_tool_call)

    def test_process_turn_end_flushes_remaining_text(self):
        self.manager.active = True
        self.manager.process_delta("This is unfinished")
        self.assertTrue(self.manager.sentence_queue.empty())
        
        self.manager.process_turn_end()
        self.assertEqual(self.manager.sentence_queue.get(timeout=1), "This is unfinished")
        self.assertEqual(self.manager.sentence_queue.get(timeout=1), "<TURN_COMPLETED>")

    def test_voice_loop_adds_trace_entries_on_recording_error(self):
        self.manager.config = VoiceConfig(keep_audio=False)
        self.manager.recorder = Mock()
        
        call_count = [0]
        
        def set_stop(*args, **kwargs):
            if call_count[0] == 0:
                call_count[0] += 1
                raise RuntimeError("mic error")
            else:
                self.manager.stop_event.set()
                return "fake_path.wav"
            
        self.manager.recorder.record_to_file.side_effect = set_stop
        
        # We also need self.manager.stt to be mocked so it doesn't crash on fake_path.wav
        self.manager.stt = Mock()
        self.manager.stt.transcribe.return_value = Mock(text="")
        
        self.manager._voice_loop()
        
        traces = self.tui.session_state.traces
        self.assertTrue(any(t["summary"] == "VAD" for t in traces))
        error_traces = [t for t in traces if t["summary"] == "Recording Error"]
        self.assertEqual(len(error_traces), 1)
        self.assertEqual(error_traces[0]["detail"], "mic error")

if __name__ == "__main__":
    unittest.main()
