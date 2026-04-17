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
from rocky.memory.manager import MemoryManager
from rocky.llm import LLM, ChatLLM, Gemma4LLM
from rocky.session import SessionState
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
                    {"text": "Earlier context: user preference and fuel shortage."},
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

        summary = manager.summarize_dialogue(dialogue)
        manager.learn(dialogue, summary=summary)

        recalled = manager.build_context("concise fuel shortage")

        self.assertEqual(len(recalled), 2)
        self.assertTrue(recalled[0].startswith("User prefers concise answers and discussed fuel shortage."))
        self.assertTrue(recalled[1].startswith("User prefers concise answers and discussed fuel shortage."))

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
        recalled = manager.build_context("no thanks")

        self.assertEqual(report["semantic"], [])
        self.assertEqual(report["episodic"], [])
        self.assertEqual(recalled, [])

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

    def test_memory_manager_persists_entries_to_sqlite(self):
        manager = MemoryManager(dialogue_window=2, db_path=self.db_path)
        self.addCleanup(manager.close)
        manager.llm = Mock(
            generate_raw=Mock(
                side_effect=[
                    {"text": "Earlier context: user prefers concise answers."},
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

        summary = manager.summarize_dialogue(dialogue)
        manager.learn(dialogue, summary=summary)
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
        recalled = reloaded.build_context("concise answers")
        self.assertEqual(len(recalled), 2)
        self.assertTrue(recalled[0].startswith("User prefers concise answers."))
        self.assertTrue(recalled[1].startswith("User prefers concise answers."))

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
        semantic_rows = db.load_semantic_documents()

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
            summarize_every=2,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.system_prompt = "sys"
        agent.memory_manager.dialogue = [user_message("seed")]
        agent.memory_manager.build_memory_routes = Mock(return_value={"semantic": False, "episodic": False})
        agent.memory_manager.build_memory_sections = Mock(return_value=("", ""))
        agent.memory_manager.summarize_dialogue = Mock(return_value="summary paragraph")
        agent.memory_manager.learn = Mock()
        agent.memory_manager.trim_dialogue = Mock()
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

        agent.memory_manager.summarize_dialogue.assert_not_called()
        agent.memory_manager.learn.assert_not_called()
        agent.memory_manager.trim_dialogue.assert_not_called()
        self.assertEqual(agent.session_state.last_answer, "ok")
        self.assertIn("assistant_message", [event.type for event in events])

    def test_agent_process_delegates_to_llm(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            summarize_every=1,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.system_prompt_template = "sys"
        agent.system_prompt = "sys"
        agent.memory_manager.dialogue = [user_message("seed"), user_message("old")]
        agent.memory_manager.build_memory_routes = Mock(return_value={"semantic": False, "episodic": False})
        agent.memory_manager.build_memory_sections = Mock(return_value=("", ""))
        agent.memory_manager.summarize_dialogue = Mock(return_value="summary paragraph")
        agent.memory_manager.learn = Mock()
        agent.memory_manager.trim_dialogue = Mock()
        captured_prompt_contexts = []

        def capture_stream(context):
            captured_prompt_contexts.append(
                {
                    "system_prompt": context.system_prompt,
                    "context": list(context.context),
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

        agent.memory_manager.summarize_dialogue.assert_called_once()
        agent.llm.generate_stream.assert_called_once()
        agent.memory_manager.build_memory_sections.assert_called_once_with(
            "hello",
            routes={"semantic": False, "episodic": False},
        )
        learn_dialogue = agent.memory_manager.learn.call_args.args[0]
        self.assertEqual([entry.content for entry in learn_dialogue], ["seed", "old", "hello", "ok"])
        self.assertEqual(agent.memory_manager.learn.call_args.kwargs["summary"], "summary paragraph")
        agent.memory_manager.trim_dialogue.assert_called_once()
        self.assertEqual(captured_prompt_contexts[0]["system_prompt"], "sys")
        self.assertEqual(captured_prompt_contexts[0]["context"], [])
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
            summarize_every=99,
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
        self.assertEqual(event_types[0], "status_changed")
        self.assertEqual(event_types[1], "user_message")
        self.assertIn("reasoning_update", event_types)
        self.assertIn("assistant_message", event_types)
        self.assertIn("routing", [entry["phase"] for entry in agent.session_state.current_trace])
        self.assertEqual(agent.session_state.last_reasoning, "Check the memory and answer directly.")
        self.assertEqual(agent.session_state.last_answer, "Here is the answer.")

    def test_agent_skips_memory_trace_when_router_returns_no_memory(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            summarize_every=99,
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
            summarize_every=99,
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
        self.assertIn("tool_call_started", event_types)
        self.assertIn("tool_call_result", event_types)
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
            summarize_every=99,
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

    def test_agent_force_compact_saves_notice(self):
        self.agent = RockyAgent(
            model="llama3.1",
            dialogue_window=1,
            summarize_every=99,
            memory_db_path=self.db_path,
        )
        agent = self.agent
        agent.memory_manager.dialogue = [
            user_message("seed"),
            assistant_message("reply"),
        ]
        agent.memory_manager.summarize_dialogue = Mock(return_value="seed summary")
        agent.memory_manager.learn = Mock()
        agent.memory_manager.trim_dialogue = Mock()

        event = agent.force_compact()

        self.assertEqual(event.type, "summary_created")
        self.assertEqual(agent.session_state.notice, "Dialogue compacted into memory.")
        self.assertIn("memory", [entry["phase"] for entry in agent.session_state.current_trace])
        self.assertIn(
            "Memory compaction complete.",
            [entry["summary"] for entry in agent.session_state.current_trace],
        )
        agent.memory_manager.learn.assert_called_once()
        agent.memory_manager.trim_dialogue.assert_called_once()


if __name__ == "__main__":
    unittest.main()
