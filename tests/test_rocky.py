import json
import os
import subprocess
import tempfile
import sys
import unittest
from unittest.mock import Mock

from rocky import RockyAgent
from rocky.conversation import PromptContext, assistant_message, tool_message, user_message
from rocky.memory.db import MemoryDB
from rocky.memory.manager import (
    MemoryManager,
    RECALL_EPISODIC_SYSTEM_PROMPT,
    RECALL_SEMANTIC_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from rocky.memory.policy import (
    EpisodicCandidate,
    MemoryPolicy,
    MemoryPolicyConfig,
    MemoryWriteCandidateSet,
    SemanticCandidate,
)
from rocky.memory.compaction import CompactionConfig, CompactionTrigger
from rocky.memory.monologue import Monologue
from rocky.memory.trigger import MemoryWritePolicy, MemoryWritePolicyConfig
from rocky.llm import LLM, ChatLLM, Gemma4LLM
from rocky.events import AgentEvent
from rocky.session import SessionState
from rocky.tracing import TraceLog
from rocky.tools.manager import ToolManager
from rocky.tools.registry import TOOLS_REGISTRY
from rocky.tui.app import (
    RockyTUI,
    build_dialogue_lines,
    build_header_lines,
    build_input_line,
    build_presence_lines,
    build_telemetry_line,
    build_thought_lines,
)


class ToolFormattingTests(unittest.TestCase):
    def setUp(self):
        self.tool_manager = ToolManager(TOOLS_REGISTRY)

    def test_extract_tool_call_parses_gemma_block(self):
        text = '<|tool_call>call:analyze_material{material:<|"|>xenonite<|"|>}<tool_call|>'

        parsed = self.tool_manager.extract_tool_call(text)

        self.assertEqual(parsed, {
            "tool": "analyze_material",
            "args": {"material": "xenonite"},
        })

    def test_format_tool_response_uses_gemma_tokens(self):
        response = self.tool_manager.format_tool_response("analyze_material", "good metal")

        self.assertIn("<|tool_response>", response)
        self.assertIn("response:analyze_material", response)
        self.assertIn("<|\"|>good metal<|\"|>", response)
        self.assertTrue(response.endswith("<tool_response|>"))

    def test_strip_thoughts_removes_thought_block(self):
        text = "<|channel>thought\ninternal note\n<channel|>Final answer."

        cleaned = self.tool_manager.strip_thoughts(text)

        self.assertEqual(cleaned, "Final answer.")

    def test_strip_thoughts_removes_meta_prefixes(self):
        text = "Internally: would Rocky say this? No. Speak plainly."

        cleaned = self.tool_manager.strip_thoughts(text)

        self.assertEqual(cleaned, "would Rocky say this? No. Speak plainly.")

    def test_prompt_section_includes_gemma_tool_declaration(self):
        section = self.tool_manager.get_prompt_section()

        self.assertIn("Gemma 4 tool declarations:", section)
        self.assertIn("<|tool>declaration:analyze_material", section)

    def test_system_prompt_blocks_meta_narration(self):
        from rocky.config import SYSTEM_PROMPT

        self.assertIn("Do not reveal internal reasoning", SYSTEM_PROMPT)
        self.assertNotIn("Before every reply ask internally:", SYSTEM_PROMPT)

    def test_system_prompt_separates_user_identity_from_rocky(self):
        from rocky.config import SYSTEM_PROMPT

        self.assertIn("## Identity Boundary", SYSTEM_PROMPT)
        self.assertIn('If asked who **they** are, or about "I", "me", "my"', SYSTEM_PROMPT)
        self.assertIn("Do not assume the user is Rocky", SYSTEM_PROMPT)

    def test_prompt_section_can_skip_gemma_declarations(self):
        section = self.tool_manager.get_prompt_section(include_declarations=False)

        self.assertNotIn("Gemma 4 tool declarations:", section)
        self.assertIn("Available tools:", section)

    def test_dialogue_lines_show_latest_exchange_without_tool_noise(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.recent_dialogue = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": '<|tool_call>call:analyze_material{material:<|"|>steel<|"|>}<tool_call|>'},
            {"role": "tool", "content": "Strong structural metal. Heavy. Good for mechanical frames."},
            {
                "role": "assistant",
                "content": (
                    "I have analyzed the material steel for you! "
                    "Steel is strong, dense, and a solid choice for mechanical frames, "
                    "bridges, and heavy machinery components."
                ),
            },
        ]

        lines = build_dialogue_lines(state, width=80, height=10)

        joined = "\n".join(lines)
        self.assertIn("EXCHANGE", joined)
        self.assertTrue(any(line.startswith("─") and "EXCHANGE" in line for line in lines))
        self.assertNotIn("<|tool_call>", joined)
        self.assertIn("You", joined)
        self.assertIn("Rocky", joined)
        self.assertIn(">", joined)
        self.assertNotIn("Tool >", joined)
        self.assertIn("I have analyzed the material steel for you!", joined)
        self.assertIn("mechanical frames,", joined)
        self.assertIn("bridges, and heavy", joined)
        self.assertIn("machinery components.", joined)

    def test_dialogue_lines_align_labels_and_add_spacing_between_messages(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.recent_dialogue = [
            {"role": "user", "content": "Hello who is this?"},
            {"role": "assistant", "content": "I am Rocky."},
            {"role": "user", "content": "Tell me more."},
            {"role": "assistant", "content": "Gladly."},
        ]

        lines = build_dialogue_lines(state, width=60, height=12)

        message_rows = [index for index, line in enumerate(lines) if line.strip().startswith(("You", "Rocky"))]
        self.assertGreaterEqual(len(message_rows), 4)
        self.assertTrue(all((right - left) > 1 for left, right in zip(message_rows, message_rows[1:])))
        self.assertEqual(
            len({line.index(">") for line in lines if line.strip().startswith(("You", "Rocky"))}),
            1,
        )

    def test_dialogue_lines_drop_older_entries_when_space_is_tight(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.recent_dialogue = [
            {"role": "user", "content": "Old question one"},
            {"role": "assistant", "content": "Old answer one"},
            {"role": "user", "content": "Old question two"},
            {"role": "assistant", "content": "Old answer two"},
            {"role": "user", "content": "Newest question"},
            {"role": "assistant", "content": "Newest answer"},
        ]

        lines = build_dialogue_lines(state, width=50, height=4)

        joined = "\n".join(lines)
        self.assertIn("Newest question", joined)
        self.assertIn("Newest answer", joined)
        self.assertNotIn("Old question one", joined)
        self.assertNotIn("Old answer one", joined)

    def test_thought_lines_include_event_stream(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.current_trace = [
            {"phase": "status", "summary": "Thinking"},
            {"phase": "memory", "summary": "loaded 1 semantic, 0 episodic memories", "detail": "ship materials"},
            {"phase": "intent", "summary": "challenge the premise"},
        ]

        lines = build_thought_lines(state, width=80, height=10)

        joined = "\n".join(lines)
        self.assertIn("THOUGHTS", joined)
        self.assertTrue(any(line.startswith("─") and "THOUGHTS" in line for line in lines))
        self.assertIn("status", joined)
        self.assertIn("Thinking", joined)
        self.assertIn("semantic memory • loaded 1 semantic, 0 episodic memories", joined)
        self.assertIn("episodic memory • ship materials", joined)
        self.assertIn("ship materials", joined)
        self.assertIn("intent", joined)
        self.assertIn("challenge the premise", joined)

    def test_thought_lines_align_phase_labels(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.current_trace = [
            {"phase": "status", "summary": "Thinking"},
            {"phase": "routing", "summary": "Semantic memory selected."},
            {"phase": "memory", "summary": "ship materials", "detail": "none"},
        ]

        lines = build_thought_lines(state, width=80, height=10)

        memory_columns = {
            line.index("•")
            for line in lines
            if line.strip().startswith(("status", "routing", "semantic memory", "episodic memory")) and "•" in line
        }
        self.assertEqual(len(memory_columns), 1)

    def test_thought_and_exchange_share_separator_column(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.current_trace = [
            {"phase": "status", "summary": "Thinking"},
            {"phase": "memory", "summary": "ship materials", "detail": "none"},
        ]
        state.recent_dialogue = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        thought_lines = build_thought_lines(state, width=80, height=8)
        exchange_lines = build_dialogue_lines(state, width=80, height=8)

        thought_columns = {line.index("•") for line in thought_lines if "•" in line}
        exchange_columns = {line.index(">") for line in exchange_lines if ">" in line}
        self.assertEqual(len(thought_columns), 1)
        self.assertEqual(len(exchange_columns), 1)
        self.assertEqual(next(iter(thought_columns)), next(iter(exchange_columns)))

    def test_thought_lines_add_spacing_between_entries(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.current_trace = [
            {"phase": "status", "summary": "Thinking"},
            {"phase": "routing", "summary": "Semantic memory selected."},
            {"phase": "intent", "summary": "challenge the premise"},
        ]

        lines = build_thought_lines(state, width=60, height=12)

        trace_rows = [
            index
            for index, line in enumerate(lines)
            if line.strip().startswith(("status", "routing", "intent"))
        ]
        self.assertGreaterEqual(len(trace_rows), 3)
        self.assertTrue(all((right - left) > 1 for left, right in zip(trace_rows, trace_rows[1:])))

    def test_memory_lines_show_semantic_and_episodic_separately(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.current_trace = [
            {
                "phase": "memory",
                "summary": "Humans are defined by intelligence, creativity, and emotional complexity.",
                "detail": "Meaning is found in connection and understanding rather than perfect logic.",
            }
        ]

        lines = build_thought_lines(state, width=80, height=10)

        joined = "\n".join(lines)
        self.assertIn("semantic memory • Humans are defined by intelligence, creativity,", joined)
        self.assertIn("Humans are defined by intelligence, creativity,", joined)
        self.assertIn("emotional complexity.", joined)
        self.assertIn("episodic memory • Meaning is found in connection and understanding", joined)
        self.assertIn("Meaning is found in connection and understanding", joined)
        self.assertIn("than perfect logic.", joined)

    def test_thought_lines_can_render_empty(self):
        state = SessionState(model_name="m", provider_kind="chat")

        lines = build_thought_lines(state, width=80, height=8)

        joined = "\n".join(lines)
        self.assertIn("THOUGHTS", joined)
        self.assertNotIn("No reasoning captured yet.", joined)
        self.assertGreaterEqual(len(lines), 1)

    def test_thought_lines_wrap_long_notice(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.current_trace = [
            {
                "phase": "status",
                "summary": "Commands available",
                "detail": "/memory /memory list [kind] [n] /memory delete [kind] [selector] /memory delete all [kind] /reset /tools /compact /clear /cancel /quit",
            }
        ]

        lines = build_thought_lines(state, width=50, height=10)

        joined = "\n".join(lines)
        self.assertIn("status", joined)
        self.assertIn("Commands available", joined)
        self.assertIn("/memory delete [kind]", joined)
        self.assertIn("[selector]", joined)
        self.assertIn("/reset", joined)
        self.assertIn("/tools", joined)
        self.assertIn("/cancel", joined)
        self.assertIn("/quit", joined)
        self.assertNotIn("┌", joined)
        self.assertNotIn("└", joined)
        self.assertNotIn("│", joined)

    def test_header_lines_show_minimal_identity(self):
        state = SessionState(model_name="m", provider_kind="chat")

        lines = build_header_lines(state, width=80)

        joined = "\n".join(lines)
        self.assertIn("R O C K Y", joined)
        self.assertNotIn("Cross-Species Communication Link Active", joined)
        self.assertNotIn("╔", joined)
        self.assertNotIn("╚", joined)
        self.assertNotIn("║", joined)
        self.assertNotIn("Tools linked:", joined)
        self.assertNotIn("Status:", joined)
        self.assertNotIn("Model:", joined)
        self.assertNotIn("Vessel:", joined)

    def test_presence_and_telemetry_render_live_status(self):
        state = SessionState(model_name="gemma4:e2b", provider_kind="gemma")
        state.status = "thinking"
        state.tool_count = 3
        state.tool_activity = "calling tool: analyze_material"

        presence = build_presence_lines(state, width=60, frame=1)
        next_presence = build_presence_lines(state, width=60, frame=2)
        telemetry = build_telemetry_line(state, width=60)

        joined_presence = "\n".join(presence)
        next_joined_presence = "\n".join(next_presence)
        self.assertIn("◆", joined_presence)
        self.assertIn("✦", joined_presence)
        self.assertNotEqual(joined_presence, next_joined_presence)
        self.assertIn("Working", telemetry)
        self.assertIn("gemma4:e2b", telemetry)
        self.assertIn("1 tool active / 3 available", telemetry)

        input_line = build_input_line("", width=60, frame=2)
        self.assertIn("transmit thought...", input_line)

        typed_line = build_input_line("this is a fairly long prompt that should stay anchored", width=30, frame=2)
        self.assertTrue(typed_line.startswith("> "))
        self.assertIn("anchored", typed_line)
        self.assertNotIn("transmit thought...", typed_line)

    def test_thought_lines_show_previous_traces(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.current_trace = [
            {"phase": "intent", "summary": "Current reasoning."},
            {"phase": "response", "summary": "final answer ready"},
        ]

        lines = build_thought_lines(state, width=80, height=12)

        joined = "\n".join(lines)
        self.assertIn("Current reasoning.", joined)
        self.assertNotIn("final answer ready", joined)

    def test_thought_lines_show_selected_semantic_titles(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.current_trace = [
            {"phase": "memory", "summary": "ship materials, hull alloys", "detail": "none"},
        ]

        lines = build_thought_lines(state, width=80, height=10)

        joined = "\n".join(lines)
        self.assertIn("semantic memory • ship materials, hull alloys", joined)
        self.assertIn("episodic memory • None", joined)

    def test_submit_prompt_sends_plain_turn(self):
        agent = Mock()
        session_state = SessionState(model_name="m", provider_kind="chat")
        agent.get_session_state = Mock(return_value=session_state)
        agent.process_turn = Mock()
        tui = RockyTUI(agent)
        tui.buffer_text = "stale"

        tui.submit_prompt("Can you help?")

        agent.process_turn.assert_called_once_with("Can you help?", on_event=tui.handle_event)
        self.assertEqual(tui.buffer_text, "")

    def test_submit_prompt_runs_slash_command_without_agent_turn(self):
        agent = Mock()
        session_state = SessionState(model_name="m", provider_kind="chat")
        agent.get_session_state = Mock(return_value=session_state)
        agent.process_turn = Mock()
        agent.tool_manager = Mock(list_tools=Mock(return_value=[{"name": "inspect", "description": "Inspect things"}]))
        agent.memory_manager = Mock(
            dialogue=["existing"],
            list_memory_titles=Mock(return_value=["Memory One"]),
            get_semantic_memory=Mock(
                return_value=Mock(
                    title="User",
                    content="User prefers concise answers.",
                    importance=7,
                    aliases=["Profile"],
                    tags=["persona"],
                )
            ),
            delete_memory=Mock(return_value={"deleted": True}),
            delete_all_memory=Mock(return_value={"deleted": True, "count": 1}),
        )
        agent.reset_session = Mock()
        agent.force_compact = Mock()
        tui = RockyTUI(agent)

        tui.submit_prompt("/tools")

        agent.process_turn.assert_not_called()
        self.assertIn("Tools listed.", tui.session_state.notice)
        self.assertEqual(len(tui.session_state.current_trace), 1)

        tui.submit_prompt("/memory list semantic 1")
        self.assertIn("Memory One", tui.session_state.current_trace[-1]["detail"])

        tui.submit_prompt("/memory search User")
        self.assertIn("User prefers concise answers.", tui.session_state.current_trace[-1]["detail"])

        tui.submit_prompt("/quit")
        self.assertTrue(tui._should_exit)

    def test_submit_prompt_memory_capture_stores_semantic_memory(self):
        agent = Mock()
        session_state = SessionState(model_name="m", provider_kind="chat")
        agent.get_session_state = Mock(return_value=session_state)
        agent.process_turn = Mock()
        agent.tool_manager = Mock(list_tools=Mock(return_value=[]))
        agent.memory_manager = Mock(
            dialogue=[],
            recent_dialogue=Mock(return_value=[]),
            snapshot=Mock(return_value={}),
            add_semantic_memory=Mock(return_value=Mock(title="Ship materials", content="Xenonite resists heat")),
        )
        tui = RockyTUI(agent)

        tui.submit_prompt("/memory Ship materials :: Xenonite resists heat")

        agent.process_turn.assert_not_called()
        agent.memory_manager.add_semantic_memory.assert_called_once_with(
            title="Ship materials",
            content="Xenonite resists heat",
        )
        self.assertIn("Stored semantic memory", tui.session_state.notice)

    def test_tui_handle_event_applies_trace_and_stream_updates_incrementally(self):
        agent = Mock()
        session_state = SessionState(model_name="m", provider_kind="chat")
        session_state.add_trace_entry(phase="routing", summary="Semantic memory selected.")
        agent.get_session_state = Mock(return_value=session_state)
        tui = RockyTUI(agent)
        tui._render = Mock()

        tui.handle_event(AgentEvent(type="trace_emitted", payload={"phase": "routing", "summary": "Semantic memory selected."}))
        tui.handle_event(AgentEvent(type="user_message", payload={"content": "hello"}))
        tui.handle_event(AgentEvent(type="assistant_delta", payload={"content": "Hi there"}))
        tui.handle_event(AgentEvent(type="reasoning_update", payload={"content": "Answer directly."}))
        tui.handle_event(AgentEvent(type="status_changed", payload={"status": "thinking", "notice": "Working"}))

        self.assertEqual(tui.session_state.current_trace[-1]["phase"], "routing")
        self.assertEqual(tui.session_state.recent_dialogue[0]["role"], "user")
        self.assertEqual(tui.session_state.recent_dialogue[-1]["content"], "Hi there")
        self.assertEqual(tui.session_state.last_reasoning, "Answer directly.")
        self.assertEqual(tui.session_state.status, "thinking")
        self.assertEqual(tui.session_state.notice, "Working")
        self.assertGreaterEqual(tui._render.call_count, 5)

    def test_tui_handle_error_event_sets_error_status(self):
        agent = Mock()
        session_state = SessionState(model_name="m", provider_kind="chat")
        agent.get_session_state = Mock(return_value=session_state)
        tui = RockyTUI(agent)
        tui._render = Mock()

        tui.handle_event(AgentEvent(type="error", payload={"message": "Failed to connect to Ollama."}))

        self.assertEqual(tui.session_state.status, "error")
        self.assertEqual(tui.session_state.notice, "Failed to connect to Ollama.")

    def test_tui_trace_event_does_not_duplicate_existing_trace_entry(self):
        agent = Mock()
        session_state = SessionState(model_name="m", provider_kind="chat")
        session_state.add_trace_entry(phase="routing", summary="Semantic memory selected.")
        agent.get_session_state = Mock(return_value=session_state)
        tui = RockyTUI(agent)
        tui._render = Mock()

        tui.handle_event(
            AgentEvent(
                type="trace_emitted",
                payload={"phase": "routing", "summary": "Semantic memory selected."},
            )
        )

        self.assertEqual(len(tui.session_state.current_trace), 1)
        self.assertEqual(tui.session_state.current_trace[0]["phase"], "routing")


class AdapterTests(unittest.TestCase):
    def test_base_llm_factory_picks_gemma(self):
        llm = LLM.build(model="gemma4:e2b")

        self.assertIsInstance(llm, Gemma4LLM)

    def test_base_llm_factory_picks_chat(self):
        llm = LLM.build(model="llama3.1")

        self.assertIsInstance(llm, ChatLLM)

    def test_gemma_adapter_builds_raw_prompt(self):
        llm = Gemma4LLM()
        system_prompt = llm.build_system_prompt("You are Rocky.\n{tools_section}", "Available tools:\n\n")
        history = [user_message("hello")]
        context = PromptContext(system_prompt=system_prompt, dialogue=history)

        prompt = llm.build_prompt(context)

        self.assertTrue(prompt.startswith("<|turn>system"))
        self.assertTrue(prompt.endswith("<|turn>model\n"))

    def test_chat_adapter_builds_messages(self):
        llm = ChatLLM(model="llama3.1")
        system_prompt = llm.build_system_prompt(
            "You are Rocky.\n{tools_section}\nIf you need a tool, respond ONLY in JSON:",
            "Available tools:\n\n",
        )
        history = [
            user_message("hello"),
            tool_message("analyze_material", "good"),
        ]
        context = PromptContext(system_prompt=system_prompt, dialogue=history)

        messages = llm.build_messages(context)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn('If you need a tool, respond ONLY in JSON:', messages[0]["content"])
        self.assertEqual(messages[2]["role"], "system")
        self.assertIn("Tool result for analyze_material: good", messages[2]["content"])

    def test_chat_adapter_generate_raw_defaults_to_thinking(self):
        client = Mock()
        client.chat.return_value = {"message": {"content": "ok", "thinking": "plan"}}
        llm = ChatLLM(model="llama3.1", client=client)
        context = PromptContext(system_prompt="sys", dialogue=[user_message("hello")])

        response = llm.generate_raw(context)

        self.assertEqual(response["text"], "ok")
        self.assertEqual(response["reasoning"], "plan")
        self.assertTrue(client.chat.call_args.kwargs["think"])

    def test_gemma_adapter_generate_raw_respects_think_flag(self):
        client = Mock()
        client.generate.return_value = {"response": "ok", "thinking": ""}
        llm = Gemma4LLM(model="gemma4:e2b", client=client)
        context = PromptContext(system_prompt="sys", dialogue=[user_message("hello")])

        llm.generate_raw(context, think=False)

        self.assertFalse(client.generate.call_args.kwargs["think"])


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory.sqlite3")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_learn_does_not_summarize_by_itself(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(
                return_value={
                    "text": json.dumps(
                        {
                            "episodic_summary": "A concise memory.",
                            "semantic_facts": ["User likes apples."],
                            "importance": 7,
                            "tags": ["preference"],
                        }
                    )
                }
            )
        )
        dialogue = [
            user_message("I like apples"),
            assistant_message("Okay"),
        ]

        manager.learn(dialogue)

        self.assertEqual(len(manager.episodic.entries), 1)
        self.assertEqual(len(manager.semantic.entries), 1)
        self.assertEqual(manager.episodic.entries[0].summary, "A concise memory.")
        self.assertEqual(manager.semantic.entries[0].title, "User likes apples.")
        self.assertEqual(manager.semantic.entries[0].content, "A concise memory.")

    def test_memory_manager_recalls_semantic_and_episodic_context(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(
                side_effect=[
                    {
                        "text": json.dumps(
                            {
                                "episodic_summary": "User prefers concise answers and discussed fuel shortage.",
                                "semantic_facts": ["User prefers concise answers."],
                                "importance": 8,
                                "tags": ["preferences", "fuel"],
                            }
                        )
                    },
                    {"text": json.dumps({"semantic": True, "episodic": True})},
                    {"text": json.dumps({"selected_titles": ["User prefers concise answers."]})},
                    {"text": json.dumps({"selected_ids": ["E1"]})},
                ]
            )
        )
        dialogue = [
            user_message("I prefer concise answers"),
            assistant_message("Understood."),
            user_message("We discussed fuel shortage yesterday"),
            assistant_message("Yes, we did."),
        ]

        manager.learn(dialogue)

        semantic_section, episodic_section = manager.build_memory_sections("concise fuel shortage")

        self.assertTrue(semantic_section.startswith("User prefers concise answers and discussed fuel shortage."))
        self.assertTrue(episodic_section.startswith("User prefers concise answers and discussed fuel shortage."))

    def test_memory_manager_reports_loaded_memory_titles(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.semantic.add_document("Rocky's origin", "Rocky comes from 40 Eridani.")
        manager.episodic.add("User prefers concise answers.", "excerpt")
        manager.llm = Mock(
            generate_raw=Mock(
                side_effect=[
                    {"text": json.dumps({"semantic": True, "episodic": False})},
                    {"text": json.dumps({"selected_titles": ["Rocky's origin"]})},
                ]
            )
        )

        report = manager.build_memory_load_report("origin")

        self.assertEqual(report["semantic"], ["Rocky's origin"])
        self.assertEqual(report["episodic"], [])

    def test_memory_manager_builds_recall_index_block(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.semantic.add_document("Rocky's origin", "Rocky comes from 40 Eridani.")
        manager.episodic.add("User prefers concise answers.", "excerpt")
        manager.llm = Mock(
            generate_raw=Mock(
                side_effect=[
                    {"text": json.dumps({"semantic": True, "episodic": False})},
                    {"text": json.dumps({"selected_titles": ["Rocky's origin"]})},
                ]
            )
        )

        semantic_index = manager.build_semantic_index_block()
        semantic_section, episodic_section = manager.build_memory_sections("origin")

        self.assertIn("Semantic index:", semantic_index)
        self.assertIn("title: Rocky's origin", semantic_index)
        self.assertIn("Rocky comes from 40 Eridani.", semantic_section)
        self.assertEqual(episodic_section, "")

    def test_memory_manager_skips_memory_for_low_signal_turns(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.semantic.add_document("Rocky's origin", "Rocky comes from 40 Eridani.")
        manager.episodic.add("User prefers concise answers.", "excerpt")
        manager.llm = Mock(
            generate_raw=Mock(return_value={"text": json.dumps({"semantic": False, "episodic": False})})
        )

        report = manager.build_memory_load_report("no thanks")
        semantic_section, episodic_section = manager.build_memory_sections("no thanks")

        self.assertEqual(report["semantic"], [])
        self.assertEqual(report["episodic"], [])
        self.assertEqual(semantic_section, "")
        self.assertEqual(episodic_section, "")

    def test_memory_manager_summarizes_long_dialogue(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(generate_raw=Mock(return_value={"text": "Earlier context: first point and reply one."}))
        dialogue = [
            user_message("First point"),
            assistant_message("Reply one"),
            user_message("Second point"),
        ]

        summary = manager.summarize_dialogue(dialogue)

        self.assertIsNotNone(summary)
        self.assertEqual(summary, "Earlier context: first point and reply one.")
        manager.llm.generate_raw.assert_called_once()
        self.assertFalse(manager.llm.generate_raw.call_args.kwargs["think"])

    def test_memory_manager_persists_entries_to_sqlite(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(
                side_effect=[
                    {
                        "text": json.dumps(
                            {
                                "episodic_summary": "User prefers concise answers.",
                                "semantic_facts": ["User prefers concise answers."],
                                "importance": 8,
                                "tags": ["preference"],
                            }
                        )
                    },
                ]
            )
        )
        dialogue = [
            user_message("I prefer concise answers"),
            assistant_message("Understood."),
        ]

        manager.learn(dialogue)
        manager.close()

        reloaded = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(reloaded.close)
        reloaded.llm = Mock(
            generate_raw=Mock(
                side_effect=[
                    {"text": json.dumps({"semantic": True, "episodic": True})},
                    {"text": json.dumps({"selected_titles": ["User prefers concise answers."]})},
                    {"text": json.dumps({"selected_ids": ["E1"]})},
                ]
            )
        )

        self.assertEqual(len(reloaded.episodic.entries), 1)
        self.assertEqual(len(reloaded.semantic.entries), 1)
        semantic_section, episodic_section = reloaded.build_memory_sections("concise answers")
        self.assertIn("User prefers concise answers.", semantic_section)
        self.assertIn("User prefers concise answers.", episodic_section)

    def test_memory_manager_snapshot_includes_working_memory(self):
        manager = MemoryManager(dialogue_window=3, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.append_user("Hello")
        manager.append_assistant("Hi")

        snapshot = manager.snapshot()

        self.assertIn("working", snapshot)
        self.assertEqual(snapshot["working"]["dialogue_window"], 3)
        self.assertEqual(snapshot["working"]["active_items"], 2)
        self.assertEqual(len(snapshot["working"]["recent_dialogue"]), 2)
        self.assertEqual(snapshot["recent_dialogue"], snapshot["working"]["recent_dialogue"])

    def test_memory_manager_persists_richer_memory_metadata(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path, session_key="session-42")
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(
                side_effect=[
                    {
                        "text": json.dumps(
                            {
                                "episodic_summary": "User explored Rocky memory architecture.",
                                "semantic_facts": [
                                    {
                                        "title": "User values lifelike AI behavior",
                                        "content": "The user wants Rocky to feel intelligent rather than recorder-like.",
                                    }
                                ],
                                "importance": 9,
                                "tags": ["memory", "architecture"],
                                "emotion": "engaged",
                                "episode_type": "technical_discussion",
                                "status": "open",
                            }
                        )
                    },
                ]
            )
        )

        manager.learn([user_message("Let's design Rocky's memory.")])
        manager.close()

        reloaded = MemoryManager(dialogue_window=2, db_path=self.db_path, session_key="session-42")
        self.addCleanup(reloaded.close)

        self.assertEqual(reloaded.episodic.entries[0].episode_type, "technical_discussion")
        self.assertEqual(reloaded.episodic.entries[0].emotion, "engaged")
        self.assertEqual(reloaded.episodic.entries[0].source_session_key, "session-42")
        self.assertEqual(reloaded.episodic.entries[0].status, "open")
        self.assertAlmostEqual(reloaded.semantic.entries[0].confidence, 0.9)

    def test_memory_manager_builds_candidates_before_persisting(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path, session_key="session-42")
        self.addCleanup(manager.close)

        candidates = manager._build_write_candidates(
            {
                "episodic_summary": "User refined Rocky memory policy.",
                "semantic_facts": [
                    {
                        "title": "User wants modular memory policy",
                        "content": "The user wants memory write policy to be easy to tweak.",
                    }
                ],
                "importance": 8,
                "tags": ["memory", "policy"],
                "emotion": "engaged",
                "episode_type": "technical_discussion",
                "status": "open",
            },
            dialogue=[user_message("Let's modularize memory policy.")],
        )

        self.assertIsNotNone(candidates.episodic)
        self.assertEqual(candidates.episodic.episode_type, "technical_discussion")
        self.assertEqual(candidates.episodic.source_session_key, "session-42")
        self.assertEqual(candidates.semantic[0].title, "User wants modular memory policy")
        self.assertAlmostEqual(candidates.semantic[0].confidence, 0.8)

    def test_memory_policy_can_filter_semantic_candidates(self):
        policy = MemoryPolicy(
            MemoryPolicyConfig(
                episodic_min_importance=5,
                semantic_min_confidence=0.75,
            )
        )

        plan = policy.evaluate(
            MemoryWriteCandidateSet(
                episodic=EpisodicCandidate(
                    summary="Useful episode",
                    excerpt="excerpt",
                    importance=6,
                ),
                semantic=[
                    SemanticCandidate(title="Keep me", content="durable", confidence=0.8),
                    SemanticCandidate(title="Drop me", content="weak", confidence=0.4),
                ],
            )
        )

        self.assertIsNotNone(plan.episodic)
        self.assertEqual([candidate.title for candidate in plan.semantic], ["Keep me"])

    def test_memory_manager_default_policy_drops_low_importance_memory(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(
                return_value={
                    "text": json.dumps(
                        {
                            "episodic_summary": "Low-signal exchange.",
                            "semantic_facts": ["Temporary preference note."],
                            "importance": 6,
                            "tags": ["weak"],
                        }
                    )
                }
            )
        )

        manager.learn([user_message("maybe"), assistant_message("okay")])

        self.assertEqual(len(manager.episodic.entries), 0)
        self.assertEqual(len(manager.semantic.entries), 0)

    def test_memory_manager_default_policy_keeps_unresolved_episode_even_if_low_importance(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(
                return_value={
                    "text": json.dumps(
                        {
                            "episodic_summary": "User left an unresolved follow-up.",
                            "semantic_facts": [],
                            "importance": 4,
                            "tags": ["followup"],
                            "status": "open",
                        }
                    )
                }
            )
        )

        manager.learn([user_message("check later"), assistant_message("I will")])

        self.assertEqual(len(manager.episodic.entries), 1)
        self.assertEqual(manager.episodic.entries[0].status, "open")

    def test_memory_manager_learn_returns_write_result_counts(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(
                return_value={
                    "text": json.dumps(
                        {
                            "episodic_summary": "Strong memory.",
                            "semantic_facts": ["Durable preference."],
                            "importance": 8,
                            "tags": ["preference"],
                        }
                    )
                }
            )
        )

        result = manager.learn([user_message("I prefer direct answers."), assistant_message("Understood.")])

        self.assertEqual(result.episodic_written, 1)
        self.assertEqual(result.semantic_written, 1)

    def test_memory_write_policy_triggers_on_explicit_request(self):
        policy = MemoryWritePolicy()

        decision = policy.evaluate(
            [user_message("Please remember this preference: I prefer direct answers.")],
        )

        self.assertTrue(decision.should_write)
        self.assertIn("explicit_memory_request", decision.matched_signals)

    def test_memory_write_policy_triggers_on_durable_preference(self):
        policy = MemoryWritePolicy()

        decision = policy.evaluate(
            [user_message("I prefer direct answers.")],
        )

        self.assertTrue(decision.should_write)
        self.assertIn("durable_preference", decision.matched_signals)

    def test_memory_write_policy_does_not_trigger_on_low_signal_turn(self):
        policy = MemoryWritePolicy()

        decision = policy.evaluate(
            [user_message("okay"), assistant_message("noted")],
        )

        self.assertFalse(decision.should_write)

    def test_compaction_trigger_fires_when_dialogue_exceeds_char_limit(self):
        trigger = CompactionTrigger(CompactionConfig(max_dialogue_chars=50))

        decision = trigger.evaluate(
            [user_message("a" * 51)],
        )

        self.assertTrue(decision.should_compact)
        self.assertEqual(decision.reason, "context_limit")

    def test_compaction_trigger_does_not_fire_below_char_limit(self):
        trigger = CompactionTrigger(CompactionConfig(max_dialogue_chars=50))

        decision = trigger.evaluate(
            [user_message("short")],
        )

        self.assertFalse(decision.should_compact)

    def test_compaction_trigger_does_not_fire_on_empty_dialogue(self):
        trigger = CompactionTrigger()

        decision = trigger.evaluate([])

        self.assertFalse(decision.should_compact)

    def test_memory_manager_forces_semantic_selection_for_named_entities(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.add_semantic_memory(
            title="Cyril and his connection to Gopi",
            content="The user revealed his name is Cyril and he is Gopi's friend.",
            aliases=["User Identity Established"],
            tags=["identity"],
        )
        manager.llm = Mock(
            generate_raw=Mock(
                side_effect=[
                    {"text": json.dumps({"semantic": False, "episodic": True})},
                    {"text": json.dumps({"selected_titles": []})},
                ]
            )
        )

        report = manager.build_memory_load_report("Who is Cyril?")

        self.assertEqual(report["semantic"], ["Cyril and his connection to Gopi"])
        self.assertEqual(report["episodic"], [])

    def test_memory_prompts_treat_user_self_reference_as_user_identity(self):
        self.assertIn('If the query refers to "I", "me", "my", or "who am I"', ROUTER_SYSTEM_PROMPT)
        self.assertIn("Do not reinterpret user self-reference as Rocky's identity.", ROUTER_SYSTEM_PROMPT)
        self.assertIn("Do not select Rocky's biography or self-identity memories", RECALL_SEMANTIC_SYSTEM_PROMPT)
        self.assertIn("prioritize earlier turns where the user stated their name", RECALL_EPISODIC_SYSTEM_PROMPT)

    def test_import_markdown_file_creates_semantic_document(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)
        markdown_path = os.path.join(self.tmpdir.name, "rocky.md")
        with open(markdown_path, "w", encoding="utf-8") as handle:
            handle.write(
                "# Rocky's materials expertise\n\n"
                "Rocky focuses on practical reasoning about materials and engineering tradeoffs.\n\n"
                "## Strengths\n"
                "Rocky compares options, explains tradeoffs, and recommends practical next steps.\n"
            )

        imported = manager.import_markdown_path(markdown_path)

        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0].title, "Rocky's materials expertise")
        self.assertIn("Rocky focuses on practical reasoning", imported[0].content)
        self.assertEqual(len(manager.semantic.entries), 1)

    def test_import_markdown_file_can_contain_multiple_titles(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)
        markdown_path = os.path.join(self.tmpdir.name, "multi.md")
        with open(markdown_path, "w", encoding="utf-8") as handle:
            handle.write(
                "# Rocky's origin\n"
                "Rocky comes from 40 Eridani.\n\n"
                "## Lore\n"
                "This heading should stay as content.\n\n"
                "# Rocky's materials expertise\n"
                "Rocky compares options and tradeoffs.\n"
            )

        imported = manager.import_markdown_path(markdown_path)

        self.assertEqual(len(imported), 2)
        self.assertEqual(imported[0].title, "Rocky's origin")
        self.assertIn("## Lore", imported[0].content)
        self.assertEqual(imported[1].title, "Rocky's materials expertise")
        self.assertIn("Rocky compares options and tradeoffs.", imported[1].content)
        self.assertEqual(len(manager.semantic.entries), 2)

    def test_cli_memory_list_prints_titles(self):
        db = MemoryDB(self.db_path)
        self.addCleanup(db.close)
        db.persist_semantic_document("Rocky's origin", "Rocky comes from 40 Eridani.", 5, [], [])
        db.persist_semantic_document(
            "Rocky's materials expertise",
            "Rocky compares options and tradeoffs.",
            5,
            [],
            [],
        )

        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rocky.py")
        env = os.environ.copy()
        env["ROCKY_MEMORY_DB"] = self.db_path
        result = subprocess.run(
            [sys.executable, script_path, "memory", "list"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertIn("Semantic memories (2):", result.stdout)
        self.assertIn("1. Rocky's origin", result.stdout)
        self.assertIn("2. Rocky's materials expertise", result.stdout)

    def test_cli_memory_search_prints_full_entry(self):
        db = MemoryDB(self.db_path)
        self.addCleanup(db.close)
        db.persist_semantic_document(
            "User",
            "User prefers concise answers.",
            7,
            ["Profile"],
            ["persona"],
        )

        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rocky.py")
        env = os.environ.copy()
        env["ROCKY_MEMORY_DB"] = self.db_path
        result = subprocess.run(
            [sys.executable, script_path, "memory", "search", "User"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertIn("Title: User", result.stdout)
        self.assertIn("Content: User prefers concise answers.", result.stdout)
        self.assertIn("Importance: 7", result.stdout)
        self.assertIn("Aliases: Profile", result.stdout)
        self.assertIn("Tags: persona", result.stdout)

    def test_entity_store_upserts_and_merges_aliases(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)

        manager.entity.upsert("Cyril", entity_type="person", aliases=["C"])
        manager.entity.upsert("Cyril", entity_type="person", aliases=["Cy"])

        record = manager.entity.get("Cyril")
        self.assertIsNotNone(record)
        self.assertIn("C", record.aliases)
        self.assertIn("Cy", record.aliases)
        self.assertEqual(record.entity_type, "person")

    def test_entity_store_lookup_by_alias(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)

        manager.entity.upsert("San Francisco", entity_type="place", aliases=["SF"])

        self.assertIsNotNone(manager.entity.get("SF"))
        self.assertIsNone(manager.entity.get("Tokyo"))

    def test_entity_store_records_relations(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)

        manager.entity.upsert("Cyril", entity_type="person")
        manager.entity.add_relation("Cyril", "Gopi", label="friend of")
        manager.entity.add_relation("Cyril", "Gopi", label="friend of")  # duplicate — ignored

        record = manager.entity.get("Cyril")
        self.assertEqual(len(record.relations), 1)
        self.assertEqual(record.relations[0].to_name, "Gopi")
        self.assertEqual(record.relations[0].label, "friend of")

    def test_entity_persists_to_db_and_reloads(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)

        manager.db.upsert_entity("Cyril", entity_type="person")
        manager.db.add_entity_relation("Cyril", "Gopi", "friend of")
        manager.close()

        reloaded = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(reloaded.close)

        record = reloaded.entity.get("Cyril")
        self.assertIsNotNone(record)
        self.assertEqual(record.entity_type, "person")
        relations = reloaded.get_entity_relations("Cyril")
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].to_name, "gopi")
        self.assertEqual(relations[0].label, "friend of")

    def test_learn_extracts_and_persists_entities(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(
                return_value={
                    "text": json.dumps(
                        {
                            "episodic_summary": "User introduced Cyril as a friend.",
                            "semantic_facts": [
                                {
                                    "title": "Cyril's connection to Gopi",
                                    "content": "Cyril is Gopi's friend.",
                                    "entity_name": "Cyril",
                                    "entity_type": "person",
                                    "relations": [{"to": "Gopi", "label": "friend of"}],
                                }
                            ],
                            "importance": 8,
                            "tags": ["people"],
                        }
                    )
                }
            )
        )

        manager.learn([user_message("My friend Cyril says hi."), assistant_message("Nice to meet him.")])

        record = manager.entity.get("Cyril")
        self.assertIsNotNone(record)
        self.assertEqual(record.entity_type, "person")
        self.assertEqual(len(record.relations), 1)
        self.assertEqual(record.relations[0].label, "friend of")

    def test_entity_unknown_type_defaults_to_person(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)

        manager.entity.upsert("Xenonite", entity_type="material")

        record = manager.entity.get("Xenonite")
        self.assertEqual(record.entity_type, "person")

    def test_semantic_index_block_includes_entity_section(self):
        manager = MemoryManager(dialogue_window=1, db_path=self.db_path)
        self.addCleanup(manager.close)

        manager.semantic.add_document("Rocky's origin", "Rocky comes from 40 Eridani.")
        manager.entity.upsert("Cyril", entity_type="person")
        manager.entity.add_relation("Cyril", "Gopi", label="friend of")

        block = manager.build_semantic_index_block()

        self.assertIn("Semantic index:", block)
        self.assertIn("Rocky's origin", block)
        self.assertIn("Known entities:", block)
        self.assertIn("Cyril (person)", block)
        self.assertIn("friend of Gopi", block)

    def test_monologue_buffer_rolls_after_max_entries(self):
        monologue = Monologue(max_entries=3)

        for i in range(5):
            monologue.add(f"thought {i}", turn_index=i)

        self.assertEqual(len(monologue.entries), 3)
        self.assertEqual(monologue.entries[0].thought, "thought 2")
        self.assertEqual(monologue.entries[-1].thought, "thought 4")

    def test_monologue_build_section_formats_entries(self):
        monologue = Monologue()
        monologue.add("The user seems uncertain.", turn_index=1)
        monologue.add("My answer was too dense.", turn_index=2)

        section = monologue.build_section()

        self.assertIn("[Turn 1] The user seems uncertain.", section)
        self.assertIn("[Turn 2] My answer was too dense.", section)

    def test_monologue_build_section_is_empty_when_no_entries(self):
        monologue = Monologue()

        self.assertEqual(monologue.build_section(), "")

    def test_monologue_latest_returns_most_recent(self):
        monologue = Monologue()
        monologue.add("first", turn_index=1)
        monologue.add("second", turn_index=2)

        self.assertEqual(monologue.latest().thought, "second")

    def test_monologue_clear_empties_entries(self):
        monologue = Monologue()
        monologue.add("thought", turn_index=1)
        monologue.clear()

        self.assertIsNone(monologue.latest())
        self.assertEqual(monologue.build_section(), "")

    def test_manager_reflect_stores_thought_in_monologue(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(return_value={"text": '{"thought": "The user wants a direct answer.", "emotion": "curious"}'})
        )
        dialogue = [
            user_message("What is the best alloy for hull plating?"),
            assistant_message("Xenonite works well under pressure."),
        ]

        thought = manager.reflect(dialogue, turn_index=1)

        self.assertEqual(thought, "The user wants a direct answer.")
        self.assertEqual(manager.monologue.latest().thought, "The user wants a direct answer.")
        self.assertEqual(manager.monologue.latest().turn_index, 1)
        self.assertEqual(manager.monologue.latest().emotion.value, "curious")

    def test_manager_reflect_returns_none_on_llm_failure(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(generate_raw=Mock(side_effect=RuntimeError("no connection")))

        result = manager.reflect([user_message("hello")], turn_index=1)

        self.assertIsNone(result)
        self.assertIsNone(manager.monologue.latest())

    def test_manager_reflect_returns_none_on_empty_dialogue(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)

        result = manager.reflect([], turn_index=1)

        self.assertIsNone(result)

    def test_build_monologue_section_proxies_to_working_memory(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.working.monologue.add("Something is unresolved.", turn_index=3)

        section = manager.build_monologue_section()

        self.assertIn("[Turn 3] Something is unresolved.", section)

    def test_system_prompt_includes_monologue_section(self):
        from rocky.config import SYSTEM_PROMPT
        from rocky.llm import LLM
        llm = LLM.build(model="llama3.1")
        thought = "[Turn 1] The user needs a concrete example."

        prompt = llm.build_system_prompt(SYSTEM_PROMPT, "", monologue=thought)

        self.assertIn("Internal State", prompt)
        self.assertIn(thought, prompt)

    def test_emotion_fsm_defaults_to_neutral(self):
        from rocky.memory.emotion import EmotionFSM, EmotionState
        fsm = EmotionFSM()
        self.assertEqual(fsm.current(), EmotionState.neutral)

    def test_emotion_fsm_transition_updates_state(self):
        from rocky.memory.emotion import EmotionFSM, EmotionState
        fsm = EmotionFSM()
        fsm.transition(EmotionState.curious)
        self.assertEqual(fsm.current(), EmotionState.curious)

    def test_emotion_fsm_clear_resets_to_neutral(self):
        from rocky.memory.emotion import EmotionFSM, EmotionState
        fsm = EmotionFSM()
        fsm.transition(EmotionState.excited)
        fsm.clear()
        self.assertEqual(fsm.current(), EmotionState.neutral)

    def test_emotion_state_parse_valid(self):
        from rocky.memory.emotion import EmotionState
        self.assertEqual(EmotionState.parse("curious"), EmotionState.curious)
        self.assertEqual(EmotionState.parse("EXCITED"), EmotionState.excited)

    def test_emotion_state_parse_invalid_falls_back_to_neutral(self):
        from rocky.memory.emotion import EmotionState
        self.assertEqual(EmotionState.parse("furious"), EmotionState.neutral)
        self.assertEqual(EmotionState.parse(""), EmotionState.neutral)

    def test_reflect_updates_emotion_fsm(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(return_value={"text": '{"thought": "Something interesting.", "emotion": "excited"}'})
        )
        dialogue = [user_message("We solved it!"), assistant_message("Amaze!")]

        manager.reflect(dialogue, turn_index=2)

        from rocky.memory.emotion import EmotionState
        self.assertEqual(manager.emotion.current(), EmotionState.excited)

    def test_reflect_fsm_stays_neutral_on_invalid_emotion(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(return_value={"text": '{"thought": "Hmm.", "emotion": "bogus"}'})
        )
        dialogue = [user_message("hello"), assistant_message("hi")]

        manager.reflect(dialogue, turn_index=1)

        from rocky.memory.emotion import EmotionState
        self.assertEqual(manager.emotion.current(), EmotionState.neutral)

    def test_build_emotion_section_returns_state_value(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        from rocky.memory.emotion import EmotionState
        manager.emotion.transition(EmotionState.concerned)

        self.assertEqual(manager.build_emotion_section(), "concerned")

    def test_reset_session_clears_emotion(self):
        agent = RockyAgent(model="llama3.1", memory_db_path=self.db_path)
        from rocky.memory.emotion import EmotionState
        agent.memory_manager.emotion.transition(EmotionState.satisfied)

        agent.reset_session()

        self.assertEqual(agent.memory_manager.emotion.current(), EmotionState.neutral)

    def test_system_prompt_includes_emotion(self):
        from rocky.config import SYSTEM_PROMPT
        from rocky.llm import LLM
        llm = LLM.build(model="llama3.1")

        prompt = llm.build_system_prompt(SYSTEM_PROMPT, "", emotion="curious")

        self.assertIn("Current emotion: curious", prompt)

    def test_monologue_entry_stores_emotion(self):
        from rocky.memory.emotion import EmotionState
        from rocky.memory.monologue import Monologue
        monologue = Monologue()
        monologue.add("Interesting problem.", turn_index=1, emotion=EmotionState.curious)

        entry = monologue.latest()
        self.assertEqual(entry.emotion, EmotionState.curious)

    def test_memory_db_persists_session_snapshot(self):
        db = MemoryDB(self.db_path)
        self.addCleanup(db.close)
        db.persist_session_snapshot(
            "default",
            {"status": "idle", "turn_index": 3},
            [{"role": "user", "content": "hello"}],
        )

        snapshot = db.load_latest_session_snapshot("default")

        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["state"]["turn_index"], 3)
        self.assertEqual(snapshot["transcript"][0]["content"], "hello")

    def test_memory_db_can_clear_session_snapshots_without_touching_semantic(self):
        db = MemoryDB(self.db_path)
        self.addCleanup(db.close)
        db.persist_session_snapshot(
            "default",
            {"status": "idle"},
            [{"role": "user", "content": "hello"}],
        )
        db.persist_semantic_document("Rocky's origin", "Rocky comes from 40 Eridani.", 5, [], [])

        deleted = db.delete_all_session_snapshots()
        snapshot = db.load_latest_session_snapshot("default")
        semantic_rows = db.load_semantic_entries()

        self.assertGreaterEqual(deleted, 1)
        self.assertIsNone(snapshot)
        self.assertEqual(len(semantic_rows), 1)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory.sqlite3")
        self.agent = None

    def tearDown(self):
        if self.agent is not None:
            self.agent.memory_manager.close()
        self.tmpdir.cleanup()

    def test_agent_does_not_compact_before_threshold(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.system_prompt = "sys"
        agent.memory_manager.dialogue = [user_message("seed")]
        agent.memory_manager.build_memory_routes = Mock(return_value={"semantic": False, "episodic": False})
        agent.memory_manager.build_memory_sections = Mock(return_value=("", ""))
        agent.memory_manager.learn = Mock()
        agent.llm.generate_stream = Mock(
            return_value=iter(
                [
                    {
                        "text": "ok",
                        "reasoning": "",
                        "raw": {},
                        "done": True,
                        "text_delta": "ok",
                        "reasoning_delta": "",
                    }
                ]
            )
        )
        agent.tool_manager.extract_tool_call = Mock(return_value=None)

        events = agent.process_turn("hello", max_turns=1)

        agent.memory_manager.learn.assert_not_called()
        self.assertEqual(agent.session_state.last_answer, "ok")
        self.assertIn("assistant_message", [event.type for event in events])

    def test_agent_process_delegates_to_llm(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.system_prompt_template = "sys"
        agent.system_prompt = "sys"
        agent.memory_manager.dialogue = [user_message("seed"), user_message("old")]
        agent.memory_manager.build_memory_routes = Mock(return_value={"semantic": False, "episodic": False})
        agent.memory_manager.build_memory_sections = Mock(return_value=("", ""))
        agent.memory_manager.learn = Mock(
            return_value=Mock(episodic_written=1, semantic_written=0, episodic_summary="summary paragraph")
        )
        agent.memory_write_policy.evaluate = Mock(return_value=Mock(should_write=True))
        captured_prompt_contexts = []

        def capture_stream(context):
            captured_prompt_contexts.append(
                {
                    "system_prompt": context.system_prompt,
                    "dialogue": list(context.dialogue),
                }
            )
            return iter(
                [
                    {
                        "text": "ok",
                        "reasoning": "",
                        "raw": {},
                        "done": True,
                        "text_delta": "ok",
                        "reasoning_delta": "",
                    }
                ]
            )

        agent.llm.generate_stream = Mock(side_effect=capture_stream)
        agent.tool_manager.extract_tool_call = Mock(return_value=None)
        events = agent.process_turn("hello", max_turns=1)

        agent.llm.generate_stream.assert_called_once()
        agent.memory_manager.build_memory_sections.assert_called_once_with(
            "hello",
            routes={"semantic": False, "episodic": False},
            report=None,
        )
        learn_dialogue = agent.memory_manager.learn.call_args.args[0]
        self.assertEqual([entry.content for entry in learn_dialogue], ["seed", "old", "hello", "ok"])
        self.assertEqual(captured_prompt_contexts[0]["system_prompt"], "sys")
        self.assertEqual([entry.content for entry in captured_prompt_contexts[0]["dialogue"]], ["seed", "old", "hello"])
        self.assertEqual(agent.memory_manager.dialogue[-1].role, "assistant")
        self.assertEqual(agent.session_state.last_answer, "ok")
        self.assertIn("assistant_message", [event.type for event in events])

    def test_agent_uses_llm_kind(self):
        self.agent = RockyAgent(model="gemma4:e2b", memory_db_path=self.db_path)
        agent = self.agent

        self.assertEqual(agent.llm.kind, "gemma")

    def test_agent_loads_prompt_text(self):
        self.agent = RockyAgent(model="gemma4:e2b", memory_db_path=self.db_path)
        agent = self.agent

        self.assertIn("You are Rocky", agent.system_prompt)
        self.assertIn("Available tools:", agent.system_prompt)
        self.assertEqual(agent.memory_manager.dialogue, [])

    def test_agent_process_turn_emits_reasoning_and_answer(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.llm.generate_raw = Mock(
            return_value={
                "text": "Here is the answer.",
                "reasoning": "Check the memory and answer directly.",
                "raw": {},
            }
        )
        agent.memory_manager.build_memory_routes = Mock(return_value={"semantic": False, "episodic": False})
        agent.tool_manager.extract_tool_call = Mock(return_value=None)

        events = agent.process_turn("hello", max_turns=1)

        event_types = [event.type for event in events]
        self.assertEqual(event_types[0], "trace_emitted")
        self.assertNotIn("status_changed", event_types)
        self.assertIn("user_message", event_types)
        self.assertIn("trace_emitted", event_types)
        self.assertIn("reasoning_update", event_types)
        self.assertIn("assistant_message", event_types)
        trace_phases = [event.payload.get("phase") for event in events if event.type == "trace_emitted"]
        self.assertIn("status", trace_phases)
        self.assertIn("routing", trace_phases)
        self.assertIn("intent", trace_phases)
        self.assertIn("routing", [entry["phase"] for entry in agent.session_state.current_trace])
        self.assertEqual(agent.session_state.last_reasoning, "Check the memory and answer directly.")
        self.assertEqual(agent.session_state.last_answer, "Here is the answer.")

    def test_agent_skips_memory_trace_when_router_returns_no_memory(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.llm.generate_raw = Mock(
            return_value={
                "text": "I am here to help.",
                "reasoning": "Answer directly.",
                "raw": {},
            }
        )
        agent.memory_manager.build_memory_routes = Mock(return_value={"semantic": False, "episodic": False})
        agent.tool_manager.extract_tool_call = Mock(return_value=None)

        agent.process_turn("hello", max_turns=1)

        phases = [entry["phase"] for entry in agent.session_state.current_trace]
        self.assertIn("routing", phases)
        self.assertNotIn("memory", phases)
        self.assertEqual(
            [entry["summary"] for entry in agent.session_state.current_trace if entry["phase"] == "routing"][0],
            "No memory selected.",
        )

    def test_agent_process_turn_records_tool_activity(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.llm.generate_raw = Mock(
            return_value={
                "text": "I should call a tool.",
                "reasoning": "Need a material lookup.",
                "raw": {},
            }
        )
        agent.tool_manager.extract_tool_call = Mock(
            return_value={"tool": "analyze_material", "args": {"material": "steel"}}
        )
        agent.tool_manager.execute_with_trace = Mock(
            return_value=(
                "Strong structural metal. Heavy.",
                {
                    "name": "analyze_material",
                    "args": {"material": "steel"},
                    "result": "Strong structural metal. Heavy.",
                },
            )
        )

        events = agent.process_turn("hello", max_turns=1)

        event_types = [event.type for event in events]
        self.assertIn("trace_emitted", event_types)
        self.assertIn("tool_event", event_types)
        tool_stages = [event.payload.get("stage") for event in events if event.type == "tool_event"]
        self.assertIn("started", tool_stages)
        self.assertIn("completed", tool_stages)
        self.assertIn("calling tool: analyze_material(material=steel)", agent.session_state.tool_activity)
        self.assertIn("Strong structural metal. Heavy.", agent.session_state.tool_activity)
        self.assertIn("tool", [entry["phase"] for entry in agent.session_state.current_trace])
        self.assertIn(
            "calling tool: analyze_material(material=steel)",
            [entry["detail"] for entry in agent.session_state.current_trace if entry["phase"] == "tool"],
        )

    def test_agent_process_turn_streams_partial_updates(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.llm.generate_stream = Mock(
            return_value=iter(
                [
                    {
                        "text_delta": "Here",
                        "reasoning_delta": "Think",
                        "done": False,
                        "raw": {},
                    },
                    {
                        "text_delta": " is the answer.",
                        "reasoning_delta": " through it.",
                        "done": True,
                        "raw": {},
                    },
                ]
            )
        )
        agent.tool_manager.extract_tool_call = Mock(return_value=None)
        agent.memory_manager.build_memory_routes = Mock(
            return_value={"semantic": False, "episodic": False}
        )
        agent.memory_manager.build_memory_load_report = Mock(
            return_value={"semantic": [], "episodic": []}
        )
        agent.memory_manager.build_memory_sections = Mock(return_value=("", ""))

        events = agent.process_turn("hello", max_turns=1)

        self.assertIn("assistant_delta", [event.type for event in events])
        self.assertEqual(agent.session_state.last_answer, "Here is the answer.")
        self.assertEqual(agent.session_state.last_reasoning, "Think through it.")

    def test_agent_reuses_memory_report_for_trace_and_prompt(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.llm.generate_stream = Mock(
            return_value=iter(
                [
                    {
                        "text_delta": "Answer",
                        "reasoning_delta": "",
                        "done": True,
                        "raw": {},
                    }
                ]
            )
        )
        report = {
            "semantic": ["Rocky's origin"],
            "episodic": ["We talked about the fuel shortage."],
        }
        agent.memory_manager.build_memory_routes = Mock(
            return_value={"semantic": True, "episodic": True}
        )
        agent.memory_manager.build_memory_load_report = Mock(return_value=report)
        agent.memory_manager.build_memory_load_summary = Mock(
            wraps=agent.memory_manager.build_memory_load_summary
        )
        agent.memory_manager.build_memory_sections = Mock(
            wraps=agent.memory_manager.build_memory_sections
        )
        agent.memory_manager._render_selected_semantic_section = Mock(return_value="semantic section")
        agent.memory_manager._render_selected_episodic_section = Mock(return_value="episodic section")
        agent.tool_manager.extract_tool_call = Mock(return_value=None)

        agent.process_turn("hello", max_turns=1)

        agent.memory_manager.build_memory_load_report.assert_called_once_with(
            "hello",
            routes={"semantic": True, "episodic": True},
        )
        agent.memory_manager.build_memory_load_summary.assert_called_once_with(
            "hello",
            routes={"semantic": True, "episodic": True},
            report=report,
        )
        agent.memory_manager.build_memory_sections.assert_called_once_with(
            "hello",
            routes={"semantic": True, "episodic": True},
            report=report,
        )

    def test_agent_skips_reflection_for_low_signal_turns_by_default(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.llm.generate_stream = Mock(
            return_value=iter(
                [
                    {
                        "text_delta": "Answer",
                        "reasoning_delta": "",
                        "done": True,
                        "raw": {},
                    }
                ]
            )
        )
        agent.memory_manager.build_memory_routes = Mock(
            return_value={"semantic": False, "episodic": False}
        )
        agent.memory_manager.build_memory_sections = Mock(return_value=("", ""))
        agent.memory_manager.reflect = Mock(return_value="private thought")
        agent.memory_write_policy.evaluate = Mock(return_value=Mock(should_write=False))
        agent.tool_manager.extract_tool_call = Mock(return_value=None)

        agent.process_turn("hello", max_turns=1)

        agent.memory_manager.reflect.assert_not_called()
        agent.memory_write_policy.evaluate.assert_called_once()

    def test_agent_reflects_for_tool_turns_in_important_only_mode(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.llm.generate_raw = Mock(
            return_value={
                "text": "I should call a tool.",
                "reasoning": "Need a material lookup.",
                "raw": {},
            }
        )
        agent.memory_manager.reflect = Mock(return_value="private thought")
        agent.memory_write_policy.evaluate = Mock(return_value=Mock(should_write=False))
        agent.tool_manager.extract_tool_call = Mock(
            return_value={"tool": "analyze_material", "args": {"material": "steel"}}
        )
        agent.tool_manager.execute_with_trace = Mock(
            return_value=(
                "Strong structural metal. Heavy.",
                {
                    "name": "analyze_material",
                    "args": {"material": "steel"},
                    "result": "Strong structural metal. Heavy.",
                },
            )
        )

        agent.process_turn("hello", max_turns=1)

        agent.memory_manager.reflect.assert_called_once_with(
            agent.memory_manager.dialogue,
            turn_index=agent.session_state.turn_index,
        )

    def test_agent_reflects_when_memory_write_is_worthwhile_in_important_only_mode(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.llm.generate_stream = Mock(
            return_value=iter(
                [
                    {
                        "text_delta": "Answer",
                        "reasoning_delta": "",
                        "done": True,
                        "raw": {},
                    }
                ]
            )
        )
        agent.memory_manager.build_memory_routes = Mock(
            return_value={"semantic": False, "episodic": False}
        )
        agent.memory_manager.build_memory_sections = Mock(return_value=("", ""))
        agent.memory_manager.reflect = Mock(return_value="private thought")
        agent.memory_manager.learn = Mock()
        agent.memory_write_policy.evaluate = Mock(return_value=Mock(should_write=True))
        agent.tool_manager.extract_tool_call = Mock(return_value=None)

        agent.process_turn("hello", max_turns=1)

        agent.memory_manager.reflect.assert_called_once_with(
            agent.memory_manager.dialogue,
            turn_index=agent.session_state.turn_index,
        )

    def test_agent_can_disable_reflection_entirely(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
            reflection_mode="off",
        )
        agent = self.agent
        agent.llm.generate_stream = Mock(
            return_value=iter(
                [
                    {
                        "text_delta": "Answer",
                        "reasoning_delta": "",
                        "done": True,
                        "raw": {},
                    }
                ]
            )
        )
        agent.memory_manager.build_memory_routes = Mock(
            return_value={"semantic": False, "episodic": False}
        )
        agent.memory_manager.build_memory_sections = Mock(return_value=("", ""))
        agent.memory_manager.reflect = Mock(return_value="private thought")
        agent.memory_write_policy.evaluate = Mock(return_value=Mock(should_write=True))
        agent.memory_manager.learn = Mock()
        agent.tool_manager.extract_tool_call = Mock(return_value=None)

        agent.process_turn("hello", max_turns=1)

        agent.memory_manager.reflect.assert_not_called()

    def test_agent_process_turn_converts_llm_connection_failure_into_error_event(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.memory_manager.build_memory_routes = Mock(side_effect=ConnectionError("Failed to connect to Ollama."))

        events = agent.process_turn("hello", max_turns=1)

        event_types = [event.type for event in events]
        self.assertIn("error", event_types)
        self.assertEqual(agent.session_state.status, "error")
        self.assertEqual(agent.session_state.notice, "Failed to connect to Ollama.")
        self.assertIn("error", [entry["phase"] for entry in agent.session_state.current_trace])

    def test_agent_force_compact_saves_notice(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.memory_manager.dialogue = [
            user_message("seed"),
            assistant_message("reply"),
        ]
        agent.memory_manager.summarize_dialogue = Mock(return_value="seed summary")
        agent.memory_manager.compact_dialogue = Mock()

        event = agent.force_compact()

        self.assertEqual(event.type, "summary_created")
        self.assertEqual(agent.session_state.notice, "Dialogue compacted into working memory.")
        self.assertIn("memory", [entry["phase"] for entry in agent.session_state.current_trace])
        self.assertIn(
            "Dialogue compacted into working memory.",
            [entry["summary"] for entry in agent.session_state.current_trace],
        )
        agent.memory_manager.summarize_dialogue.assert_called_once()
        agent.memory_manager.compact_dialogue.assert_called_once_with("seed summary")


class SessionStateTests(unittest.TestCase):
    def test_trace_log_tracks_current_and_history(self):
        trace = TraceLog(entry_limit=2, history_limit=2)

        trace.add_entry("status", "Thinking", turn_index=1)
        trace.add_entry("memory", "semantic", detail="episodic", turn_index=1)
        trace.add_entry("intent", "answer directly", turn_index=1)
        trace.commit_current(turn_index=1)

        self.assertEqual(len(trace.current()), 2)
        self.assertEqual(trace.current()[0]["phase"], "memory")
        self.assertEqual(trace.current()[1]["phase"], "intent")
        self.assertEqual(len(trace.history()), 1)
        self.assertEqual(trace.history()[0]["turn_index"], 1)
        self.assertEqual(trace.history()[0]["entries"][0]["phase"], "memory")

    def test_session_state_restore_snapshot_and_export_are_core_friendly(self):
        state = SessionState(model_name="m", provider_kind="chat")

        state.restore_payload(
            {
                "status": "thinking",
                "last_reasoning": "trace it",
                "last_answer": "answer",
                "notice": "working",
                "tool_activity": "calling tool",
                "turn_index": 4,
                "active_tool": "inspect",
                "current_trace": [{"phase": "status", "summary": "Thinking"}],
                "trace_history": [{"turn_index": 3, "entries": [{"phase": "memory", "summary": "loaded"}]}],
                "tool_history": [{"name": "inspect", "result": "ok"}],
                "memory_integrity": 91,
            }
        )
        state.sync_memory_view(
            snapshot={"integrity": 88, "semantic": [], "episodic": []},
            recent_dialogue=[{"role": "user", "content": "hello"}],
        )

        exported = state.export()

        self.assertEqual(state.status, "thinking")
        self.assertEqual(state.active_tool, "inspect")
        self.assertEqual(state.memory_integrity, 88)
        self.assertEqual(state.recent_dialogue[0]["content"], "hello")
        self.assertEqual(exported["interaction"]["turn_index"], 4)
        self.assertEqual(exported["tooling"]["active_tool"], "inspect")
        self.assertEqual(exported["memory_view"]["integrity"], 88)
        self.assertEqual(exported["trace"]["current"][0]["phase"], "status")

    def test_session_state_trace_properties_proxy_to_trace_log(self):
        state = SessionState(model_name="m", provider_kind="chat")

        state.current_trace = [{"phase": "status", "summary": "Thinking"}]
        state.trace_history = [{"turn_index": 1, "entries": [{"phase": "memory", "summary": "loaded"}]}]

        self.assertEqual(state.current_trace[0]["phase"], "status")
        self.assertEqual(state.trace.current()[0]["phase"], "status")
        self.assertEqual(state.trace_history[0]["turn_index"], 1)
        self.assertEqual(state.trace.history()[0]["entries"][0]["phase"], "memory")

    def test_session_state_reset_runtime_clears_runtime_without_requiring_tui(self):
        state = SessionState(model_name="m", provider_kind="chat")
        state.status = "thinking"
        state.turn_index = 7
        state.last_reasoning = "reasoning"
        state.last_answer = "answer"
        state.notice = "busy"
        state.tool_activity = "calling tool"
        state.active_tool = "inspect"
        state.tool_history = [{"name": "inspect"}]
        state.recent_dialogue = [{"role": "assistant", "content": "hello"}]
        state.current_trace = [{"phase": "status", "summary": "Thinking"}]
        state.trace_history = [{"turn_index": 6, "entries": [{"phase": "memory", "summary": "loaded"}]}]

        state.reset_runtime(notice="Reset complete.")

        self.assertEqual(state.status, "idle")
        self.assertEqual(state.turn_index, 0)
        self.assertEqual(state.notice, "Reset complete.")
        self.assertEqual(state.tool_history, [])
        self.assertEqual(state.recent_dialogue, [])
        self.assertEqual(state.current_trace, [])
        self.assertEqual(state.trace_history, [])


if __name__ == "__main__":
    unittest.main()
