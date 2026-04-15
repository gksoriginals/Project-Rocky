from __future__ import annotations

from rocky.config import DEFAULT_DB_PATH, SYSTEM_PROMPT
from rocky.conversation import PromptContext
from rocky.events import AgentEvent
from rocky.llm import LLM
from rocky.memory.manager import MemoryManager
from rocky.session import SessionState
from rocky.tools.manager import ToolManager
from rocky.tools.registry import TOOLS_REGISTRY


class RockyAgent:
    def __init__(
        self,
        model=None,
        tools=TOOLS_REGISTRY,
        dialogue_window=6,
        summarize_every=4,
        memory_db_path=DEFAULT_DB_PATH,
    ):
        self.session_key = "default"
        self.summarize_every = summarize_every
        self.turns_since_summary = 0
        self.llm = LLM.build(model=model)
        self.memory_manager = MemoryManager(
            dialogue_window=dialogue_window,
            llm=self.llm,
            db_path=memory_db_path,
        )
        self.tool_manager = ToolManager(tools)
        self.session_state = SessionState(
            model_name=self.llm.model,
            provider_kind=self.llm.kind,
            tool_count=self.tool_manager.tool_count(),
        )
        self.system_prompt_template = SYSTEM_PROMPT
        self.tools_section = self.tool_manager.get_prompt_section(
            include_declarations=self.llm.kind == "gemma"
        )
        self.system_prompt = self.llm.build_system_prompt(
            self.system_prompt_template,
            self.tools_section,
        )
        self._restore_session_snapshot()

    def build_prompt_context(self, query, routes=None):
        semantic_memory, episodic_memory = self.memory_manager.build_memory_sections(query, routes=routes)
        system_prompt = self.llm.build_system_prompt(
            self.system_prompt_template,
            self.tools_section,
            semantic_memory=semantic_memory,
            episodic_memory=episodic_memory,
        )
        return PromptContext(
            system_prompt=system_prompt,
            context=[],
            dialogue=self.memory_manager.dialogue,
        )

    def get_session_state(self):
        self.session_state.tool_count = self.tool_manager.tool_count()
        self.session_state.recent_dialogue = self.memory_manager.recent_dialogue()
        self.session_state.set_memory_snapshot(self.memory_manager.snapshot())
        return self.session_state

    def _record_trace(
        self,
        phase: str,
        summary: str,
        detail: str = "",
        **metadata,
    ) -> None:
        self.session_state.add_trace_entry(
            phase=phase,
            summary=summary,
            detail=detail,
            turn_index=self.session_state.turn_index,
            **metadata,
        )

    def _restore_session_snapshot(self):
        snapshot = self.memory_manager.db.load_latest_session_snapshot(self.session_key)
        if snapshot is None:
            self.session_state.tool_count = self.tool_manager.tool_count()
            return

        transcript = snapshot.get("transcript") or []
        self.memory_manager.dialogue = []
        for item in transcript:
            role = str(item.get("role") or "user")
            content = str(item.get("content") or "")
            tool_name = item.get("tool_name")
            if role == "assistant":
                self.memory_manager.append_assistant(content)
            elif role == "tool":
                self.memory_manager.append_tool(str(tool_name or ""), content)
            else:
                self.memory_manager.append_user(content)

        state = snapshot.get("state") or {}
        if isinstance(state, dict):
            self.session_state.status = str(state.get("status") or "idle")
            self.session_state.last_reasoning = str(state.get("last_reasoning") or "")
            self.session_state.last_answer = str(state.get("last_answer") or "")
            self.session_state.notice = str(state.get("notice") or "")
            self.session_state.tool_activity = str(state.get("tool_activity") or "")
            self.session_state.turn_index = int(state.get("turn_index") or 0)
            self.session_state.active_tool = state.get("active_tool") or None
            current_trace = state.get("current_trace") or []
            if isinstance(current_trace, list):
                self.session_state.current_trace = [
                    item for item in current_trace if isinstance(item, dict)
                ]
            trace_history = state.get("trace_history") or []
            if isinstance(trace_history, list):
                self.session_state.trace_history = [
                    item for item in trace_history if isinstance(item, dict)
                ]
            tool_history = state.get("tool_history") or []
            if isinstance(tool_history, list):
                self.session_state.tool_history = [
                    item for item in tool_history if isinstance(item, dict)
                ]

        self.session_state.recent_dialogue = self.memory_manager.recent_dialogue()
        self.session_state.set_memory_snapshot(self.memory_manager.snapshot())

    def _snapshot_session_state(self) -> dict[str, object]:
        return {
            "status": self.session_state.status,
            "last_reasoning": self.session_state.last_reasoning,
            "last_answer": self.session_state.last_answer,
            "notice": self.session_state.notice,
            "tool_activity": self.session_state.tool_activity,
            "turn_index": self.session_state.turn_index,
            "active_tool": self.session_state.active_tool,
            "current_trace": list(self.session_state.current_trace),
            "trace_history": list(self.session_state.trace_history),
            "tool_history": list(self.session_state.tool_history),
            "memory_integrity": self.session_state.memory_integrity,
        }

    def save_session_snapshot(self):
        self.memory_manager.db.persist_session_snapshot(
            self.session_key,
            self._snapshot_session_state(),
            self.memory_manager.recent_dialogue(limit=len(self.memory_manager.dialogue)),
        )

    def reset_session(self):
        self.memory_manager.db.delete_all_session_snapshots()
        self.memory_manager.delete_all_memory("episodic")
        self.memory_manager.dialogue = []
        self.turns_since_summary = 0

        self.session_state.status = "idle"
        self.session_state.tool_activity = ""
        self.session_state.tool_history = []
        self.session_state.active_tool = None
        self.session_state.turn_index = 0
        self.session_state.last_reasoning = ""
        self.session_state.last_answer = ""
        self.session_state.notice = "Session reset."
        self.session_state.recent_dialogue = []
        self.session_state.clear_current_trace()
        self.session_state.trace_history = []
        self.session_state.set_memory_snapshot(self.memory_manager.snapshot())

        return AgentEvent(
            type="status_changed",
            payload={"status": "idle", "notice": "Session reset."},
        )

    def _emit_status(self, status: str) -> AgentEvent:
        self.session_state.update_status(status)
        return AgentEvent(type="status_changed", payload={"status": status})

    def _format_memory_routes(self, routes: dict[str, bool]) -> str:
        semantic = bool(routes.get("semantic"))
        episodic = bool(routes.get("episodic"))
        if semantic and episodic:
            label = "Semantic and episodic memory selected."
        elif semantic:
            label = "Semantic memory selected."
        elif episodic:
            label = "Episodic memory selected."
        else:
            label = "No memory selected."
        return label

    def _format_tool_activity(self, tool_call: dict[str, object], result: str | None = None) -> str:
        tool_name = str(tool_call.get("tool") or "tool")
        args = tool_call.get("args") or {}
        if isinstance(args, dict) and args:
            arg_text = ", ".join(f"{key}={value}" for key, value in args.items())
            activity = f"calling tool: {tool_name}({arg_text})"
        else:
            activity = f"calling tool: {tool_name}"
        if result:
            result_text = str(result).strip()
            if len(result_text) > 120:
                result_text = f"{result_text[:117].rstrip()}..."
            activity = f"{activity}; result: {result_text}"
        return activity

    def _dispatch_event(self, event: AgentEvent, events: list[AgentEvent], on_event=None) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    def _generate_turn_response(self, context: PromptContext) -> dict[str, object]:
        raw_response = self.llm.generate_raw(context)

        text = str(raw_response.get("text") or "")
        reasoning = str(raw_response.get("reasoning") or "")
        return {
            "text": self.tool_manager.strip_thoughts(text).strip(),
            "reasoning": reasoning.strip(),
            "raw": raw_response.get("raw", {}),
        }

    def _generate_turn_response_stream(self, context: PromptContext):
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        last_payload: dict[str, object] = {}

        try:
            for chunk in self.llm.generate_stream(context):
                text_delta = str(chunk.get("text_delta") or "")
                reasoning_delta = str(chunk.get("reasoning_delta") or "")
                if text_delta:
                    text_parts.append(text_delta)
                if reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                last_payload = dict(chunk.get("raw") or {})
                yield {
                    "text": "".join(text_parts),
                    "reasoning": "".join(reasoning_parts),
                    "raw": last_payload,
                    "done": bool(chunk.get("done")),
                    "text_delta": text_delta,
                    "reasoning_delta": reasoning_delta,
                }
        except Exception:
            yield self._generate_turn_response(context)
            return

        if not text_parts and not reasoning_parts:
            yield self._generate_turn_response(context)

    def _record_tool_event(self, tool_trace: dict[str, object]) -> None:
        self.session_state.add_tool_event(tool_trace)
        self.session_state.active_tool = str(tool_trace.get("name") or "")

    def _record_turn_completion(self):
        self.session_state.commit_trace_history()
        self.session_state.recent_dialogue = self.memory_manager.recent_dialogue()
        self.session_state.set_memory_snapshot(self.memory_manager.snapshot())
        self.save_session_snapshot()

    def _compact_if_needed(self):
        self.turns_since_summary += 1
        if self.turns_since_summary < self.summarize_every:
            return None

        summary = self.memory_manager.summarize_dialogue(self.memory_manager.dialogue)
        if summary is not None:
            self.session_state.set_notice("Writing episodic/semantic memory...")
            self._record_trace("memory", "Writing episodic memory...")
            self._record_trace("memory", "Writing semantic memory...")
            self.memory_manager.learn(self.memory_manager.dialogue, summary=summary)
            self.memory_manager.trim_dialogue()
            self.turns_since_summary = 0
            self.session_state.set_memory_snapshot(self.memory_manager.snapshot())
            self._record_trace("memory", "Memory compaction complete.")
            return AgentEvent(type="summary_created", payload={"summary": summary})

        self.turns_since_summary = 0
        return None

    def force_compact(self):
        summary = self.memory_manager.summarize_dialogue(self.memory_manager.dialogue)
        if summary is None:
            self.session_state.set_notice("No dialogue available to compact.")
            return AgentEvent(type="status_changed", payload={"status": self.session_state.status})

        self._record_trace("memory", "Writing episodic memory...")
        self._record_trace("memory", "Writing semantic memory...")
        self.memory_manager.learn(self.memory_manager.dialogue, summary=summary)
        self.memory_manager.trim_dialogue()
        self.turns_since_summary = 0
        self.session_state.set_notice("Dialogue compacted into memory.")
        self._record_trace("memory", "Memory compaction complete.")
        self.session_state.set_memory_snapshot(self.memory_manager.snapshot())
        self.session_state.recent_dialogue = self.memory_manager.recent_dialogue()
        self.save_session_snapshot()
        return AgentEvent(type="summary_created", payload={"summary": summary})

    def process_turn(self, user_input: str, max_turns: int = 5, on_event=None):
        events: list[AgentEvent] = []
        self.session_state.turn_index += 1
        self.session_state.set_tool_activity("")
        self.session_state.clear_current_trace()
        self._dispatch_event(self._emit_status("thinking"), events, on_event)
        self._record_trace("status", "Thinking")
        self.memory_manager.append_user(user_input)
        self._dispatch_event(
            AgentEvent(type="user_message", payload={"content": user_input}),
            events,
            on_event,
        )

        for _ in range(max_turns):
            response = None
            routes = self.memory_manager.build_memory_routes(user_input)
            route_activity = self._format_memory_routes(routes)
            self.session_state.set_notice(route_activity)
            self._record_trace("routing", route_activity)
            if routes["semantic"] or routes["episodic"]:
                memory_load = self.memory_manager.build_memory_load_summary(user_input, routes=routes)
                self._record_trace(
                    "memory",
                    memory_load["semantic"],
                    memory_load["episodic"],
                )
            prompt_context = self.build_prompt_context(user_input, routes=routes)
            saw_reasoning_delta = False
            saw_text_delta = False
            for response in self._generate_turn_response_stream(prompt_context):
                reasoning = str(response.get("reasoning") or "").strip()
                assistant_text = str(response.get("text") or "").strip()
                reasoning_delta = str(response.get("reasoning_delta") or "").strip()
                text_delta = str(response.get("text_delta") or "").strip()

                if reasoning_delta:
                    saw_reasoning_delta = True
                    self.session_state.set_reasoning(reasoning)
                    self._record_trace("intent", reasoning_delta)
                    self._dispatch_event(
                        AgentEvent(type="reasoning_update", payload={"content": reasoning}),
                        events,
                        on_event,
                    )

                if text_delta:
                    saw_text_delta = True
                    self.session_state.set_answer(assistant_text)
                    self._dispatch_event(
                        AgentEvent(type="assistant_delta", payload={"content": assistant_text}),
                        events,
                        on_event,
                    )

                if response.get("done"):
                    break

            if response is None:
                break

            reasoning = str(response.get("reasoning") or "").strip()
            assistant_text = str(response.get("text") or "").strip()
            if reasoning:
                self.session_state.set_reasoning(reasoning)
                self._record_trace("intent", reasoning)
                if not saw_reasoning_delta:
                    self._dispatch_event(
                        AgentEvent(type="reasoning_update", payload={"content": reasoning}),
                        events,
                        on_event,
                    )
            if assistant_text:
                self.memory_manager.append_assistant(assistant_text)
            self.session_state.set_answer(assistant_text)
            if assistant_text and not saw_text_delta:
                self._dispatch_event(
                    AgentEvent(type="assistant_delta", payload={"content": assistant_text}),
                    events,
                    on_event,
                )
            self._dispatch_event(
                AgentEvent(type="assistant_message", payload={"content": assistant_text}),
                events,
                on_event,
            )

            tool_call = self.tool_manager.extract_tool_call(assistant_text)
            if not tool_call or "tool" not in tool_call:
                break

            tool_activity = self._format_tool_activity(tool_call)
            self.session_state.set_tool_activity(tool_activity)
            self.session_state.set_notice(tool_activity)
            self._record_trace(
                "tool",
                f"{tool_call.get('tool')} started",
                tool_activity,
            )
            self._dispatch_event(
                AgentEvent(type="tool_call_started", payload={"tool_call": tool_call}),
                events,
                on_event,
            )
            tool_result, tool_trace = self.tool_manager.execute_with_trace(tool_call)
            self._record_tool_event(tool_trace)
            tool_result_activity = self._format_tool_activity(tool_call, tool_result)
            self.session_state.set_tool_activity(tool_result_activity)
            self.session_state.set_notice(tool_result_activity)
            self._record_trace(
                "tool",
                f"{tool_call.get('tool')} completed",
                tool_result_activity,
            )
            self._dispatch_event(
                AgentEvent(
                    type="tool_call_result",
                    payload={
                        "tool_call": tool_call,
                        "result": tool_result,
                        "trace": tool_trace,
                    },
                ),
                events,
                on_event,
            )
            self.memory_manager.append_tool(tool_call["tool"], tool_result)

        summary_event = self._compact_if_needed()
        if summary_event is not None:
            self._dispatch_event(summary_event, events, on_event)
        self.session_state.update_status("idle")
        self.session_state.set_notice("Turn complete.")
        self._record_trace("status", "Turn complete.")
        self._record_turn_completion()
        self._dispatch_event(self._emit_status("idle"), events, on_event)
        self._dispatch_event(
            AgentEvent(type="memory_snapshot_updated", payload=self.session_state.memory_snapshot),
            events,
            on_event,
        )
        return events

    # ------------------------
    # CLI LOOP
    # ------------------------
    def run(self):
        print(f"Rocky Agent running on {self.llm.model} [{self.llm.kind}]")
        print("Type exit to quit\n")

        while True:
            user = input("You: ").strip()

            if user.lower() in ["exit", "quit"]:
                break

            self.process_turn(user)
            reply = self.session_state.last_answer
            print("Rocky:", reply)
            print()
