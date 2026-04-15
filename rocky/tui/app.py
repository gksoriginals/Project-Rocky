from __future__ import annotations

import curses
import re
import textwrap
from rocky.agent import RockyAgent
from rocky.session import SessionState

HEADER_TITLE = "R O C K Y"

PAIR_HEADER = 1
PAIR_USER = 2
PAIR_ASSISTANT = 3
PAIR_THOUGHTS = 4
PAIR_FOOTER = 5
PAIR_MUTED = 6
PAIR_PRESENCE = 7
PAIR_TELEMETRY = 8

TOOL_CALL_BLOCK_RE = re.compile(r"<\|tool_call\>(.*?)<tool_call\|>", re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
LABEL_COLUMN_WIDTH = 15


def _wrap(text: str, width: int) -> list[str]:
    if width <= 1:
        return [text[: max(width, 0)]]
    wrapped = textwrap.wrap(text or "", width=width, replace_whitespace=False, drop_whitespace=False)
    return wrapped or [""]


def _pad(line: str, width: int) -> str:
    return line[:width].ljust(width)


def _center(line: str, width: int) -> str:
    return _pad(line.center(width), width)


def _left(line: str, width: int) -> str:
    return _pad(line.ljust(width), width)


def _separator(title: str, width: int) -> str:
    width = max(width, 4)
    label = f" {title} "
    if len(label) >= width:
        return _pad(label[:width], width)
    remaining = width - len(label)
    left = remaining // 2
    right = remaining - left
    return f"{'─' * left}{label}{'─' * right}"


def build_header_lines(state: SessionState, width: int, height: int | None = None) -> list[str]:
    return [_left(HEADER_TITLE, width)]


def _format_tool_call(text: str) -> str | None:
    match = TOOL_CALL_BLOCK_RE.search(text or "")
    if not match:
        return None

    payload = match.group(1).strip()
    if payload.startswith("call:"):
        payload = payload[len("call:") :]

    tool_name, _, args_blob = payload.partition("{")
    tool_name = tool_name.strip()
    args_blob = args_blob.rstrip("}").strip()
    if not tool_name:
        return "Rocky > [tool call]"
    if args_blob:
        args_blob = WHITESPACE_RE.sub(" ", args_blob)
        return f"Rocky > [tool: {tool_name}({args_blob})]"
    return f"Rocky > [tool: {tool_name}]"


def _format_two_column_block(
    label: str,
    contents: list[str],
    width: int,
    label_width: int,
    separator: str = " • ",
) -> list[str]:
    available = max(width - 4, 20)
    min_content_width = 10
    max_label_width = max(1, available - len(separator) - min_content_width)
    label_col_width = max(1, min(label_width, max_label_width))
    content_width = max(available - label_col_width - len(separator), min_content_width)

    label_lines = _wrap(label or "[empty]", label_col_width)
    content_source = contents or ["[empty]"]
    content_lines: list[str] = []
    for content in content_source:
        content_lines.extend(_wrap(content or "[empty]", content_width))

    row_count = max(len(label_lines), len(content_lines))
    lines: list[str] = []
    for index in range(row_count):
        label_part = label_lines[index] if index < len(label_lines) else ""
        content_part = content_lines[index] if index < len(content_lines) else ""
        if index == 0:
            lines.append(f"{label_part.ljust(label_col_width)}{separator}{content_part}")
            continue
        if label_part:
            lines.append(f"{label_part.ljust(label_col_width)}{separator}{content_part}")
        else:
            lines.append(f"{' ' * (label_col_width + len(separator))}{content_part}")
    return lines


def _format_dialogue_entry(entry: dict[str, str], width: int) -> list[str]:
    role = (entry.get("role") or "").lower()
    content = entry.get("content") or ""

    if role == "tool":
        return []

    if role == "assistant" and _format_tool_call(content):
        return []

    label = "You" if role == "user" else "Rocky" if role == "assistant" else role.title() or "Entry"
    return _format_two_column_block(
        label,
        [content.strip() or "[empty]"],
        width,
        label_width=LABEL_COLUMN_WIDTH,
        separator=" > ",
    )


def _group_dialogue_turns(entries: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    turns: list[list[dict[str, str]]] = []
    current_turn: list[dict[str, str]] = []

    for entry in entries:
        role = (entry.get("role") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        if role == "user" and current_turn:
            turns.append(current_turn)
            current_turn = []
        current_turn.append(entry)

    if current_turn:
        turns.append(current_turn)
    return turns


def _format_dialogue_turn(turn: list[dict[str, str]], width: int) -> list[str]:
    lines: list[str] = []
    for entry in turn:
        entry_lines = _format_dialogue_entry(entry, width)
        if not entry_lines:
            continue
        if lines:
            lines.append(_pad("", width))
        lines.extend(entry_lines)
    return lines


def _format_trace_entry(entry: dict[str, str], width: int) -> list[str]:
    phase = (entry.get("phase") or "trace").strip() or "trace"
    summary = (entry.get("summary") or "").strip()
    detail = (entry.get("detail") or "").strip()
    if phase.lower() == "memory":
        semantic_value = "None" if not summary or summary.lower() == "none" else summary
        episodic_value = "None" if not detail or detail.lower() == "none" else detail
        lines = _format_two_column_block("semantic memory", [semantic_value], width, label_width=LABEL_COLUMN_WIDTH)
        lines.append(_pad("", width))
        lines.extend(_format_two_column_block("episodic memory", [episodic_value], width, label_width=LABEL_COLUMN_WIDTH))
        return lines

    content_lines = [summary or detail or "[empty]"]
    if detail and detail != summary:
        content_lines.append(detail)
    return _format_two_column_block(phase, content_lines, width, label_width=LABEL_COLUMN_WIDTH)


def _fit_recent_blocks(
    title: str,
    entries: list[dict[str, str]],
    formatter,
    width: int,
    height: int,
    block_gap: int = 0,
) -> list[str]:
    height = max(height, 1)
    content_limit = max(height - 1, 0)
    blocks: list[list[str]] = []
    used = 0

    for entry in reversed(entries):
        block = formatter(entry, width)
        if not block:
            continue
        gap = block_gap if blocks else 0
        if used + gap + len(block) <= content_limit:
            if gap:
                blocks.append([_pad("", width)] * gap)
                used += gap
            blocks.append(block)
            used += len(block)
            continue
        if used == 0 and content_limit > 0:
            clipped = block[-content_limit:]
            if clipped:
                if len(block) > len(clipped):
                    clipped = ["…"] + clipped[1:] if len(clipped) > 1 else ["…"]
                blocks.append(clipped)
                used = len(clipped)
            break

    lines: list[str] = [_separator(title, width)]
    for block in reversed(blocks):
        lines.extend(block)

    while len(lines) < height:
        lines.append(_pad("", width))
    return lines[:height]


def build_thought_lines(state: SessionState, width: int, height: int) -> list[str]:
    trace = [entry for entry in state.current_trace if (entry.get("phase") or "").strip().lower() != "response"]
    return _fit_recent_blocks("THOUGHTS", trace, _format_trace_entry, width, height, block_gap=1)


def build_dialogue_lines(state: SessionState, width: int, height: int) -> list[str]:
    recent = [
        entry
        for entry in state.recent_dialogue
        if (entry.get("role") or "").lower() in {"user", "assistant"}
    ]
    turns = _group_dialogue_turns(recent)
    return _fit_recent_blocks("EXCHANGE", turns, _format_dialogue_turn, width, height, block_gap=1)


def build_presence_lines(state: SessionState, width: int, frame: int = 0) -> list[str]:
    status = (state.status or "idle").lower()
    rows = [
        [0, "       ", 1],
        [2, "         ", 3, "        ", 4],
        [5, "   ", 6],
        [7, "               ", 8],
    ]
    active_slot = frame % 9
    trail_slot = (active_slot - 1) % 9
    spark_slot = (active_slot + 1) % 9
    secondary_slot = (active_slot + 4) % 9

    def glyph(slot: int) -> str:
        if status == "error":
            return "⊗"
        if slot == active_slot:
            return "◆" if (state.tool_activity or status == "thinking") else "◉"
        if slot == trail_slot:
            return "✦" if (state.tool_activity or status == "thinking") else "⋄"
        if slot == spark_slot and (state.tool_activity or status == "thinking"):
            return "✧"
        if slot == secondary_slot and status == "thinking":
            return "◉"
        return "◇"

    rendered_rows: list[str] = []
    for row in rows:
        parts: list[str] = []
        for item in row:
            if isinstance(item, int):
                parts.append(glyph(item))
            else:
                parts.append(item)
        rendered_rows.append("".join(parts))
    return [_left(row, width) for row in rendered_rows]


def build_telemetry_line(state: SessionState, width: int) -> str:
    status = (state.status or "idle").lower()
    model_name = state.model_name or "unknown-model"
    if status == "error":
        label = "Interrupted"
    elif state.tool_activity:
        active = 1 if (state.active_tool or state.tool_activity) else 0
        label = f"Working • {model_name} • {active} tool active / {state.tool_count} available"
    elif status == "thinking":
        label = "Thinking"
    else:
        label = "Ready"

    if label.startswith("Working"):
        return _left(label, width)
    return _left(f"{label} • {model_name} • {state.tool_count} tools available", width)


def build_input_line(buffer_text: str, width: int, prompt_label: str = "transmit thought...", frame: int = 0) -> str:
    prefix = "> "
    if buffer_text:
        visible_width = max(width - len(prefix), 0)
        if len(buffer_text) > visible_width and visible_width > 1:
            text = buffer_text[-(visible_width - 1):]
            prompt = f"{prefix}…{text}"
        else:
            prompt = f"{prefix}{buffer_text}"
        return _pad(prompt, width)

    prompt = f"{prefix}{prompt_label}"
    return _pad(prompt, width)


class RockyTUI:
    def __init__(self, agent: RockyAgent):
        self.agent = agent
        self.session_state = agent.get_session_state()
        self.buffer_text = ""
        self._screen = None
        self._should_exit = False
        self._colors_ready = False
        self._frame = 0

    def run(self) -> None:
        curses.wrapper(self._run_curses)

    def submit_prompt(self, prompt: str) -> None:
        prompt = (prompt or "").strip()
        if not prompt:
            return

        if prompt.startswith("/"):
            self._run_command(prompt)
            self.session_state = self.agent.get_session_state()
            self._render()
            return

        self.buffer_text = ""
        self._render()
        self.agent.process_turn(prompt, on_event=self.handle_event)
        self.session_state = self.agent.get_session_state()
        self._render()

    def handle_event(self, event) -> None:
        self.session_state = self.agent.get_session_state()
        self._render()

    def _run_command(self, prompt: str) -> None:
        command, _, remainder = prompt[1:].strip().partition(" ")
        command = command.lower()
        remainder = remainder.strip()

        if command in {"quit", "exit"}:
            self.session_state.set_notice("Session ended.")
            self.session_state.clear_current_trace()
            self.session_state.add_trace_entry(
                phase="command",
                summary="/quit",
                detail="session exit requested",
                turn_index=self.session_state.turn_index,
            )
            self._should_exit = True
            return

        if command == "clear":
            self.agent.memory_manager.dialogue = []
            self.session_state.recent_dialogue = []
            self.session_state.last_answer = ""
            self.session_state.set_notice("Conversation cleared.")
            self.session_state.clear_current_trace()
            self.session_state.add_trace_entry(
                phase="command",
                summary="/clear",
                detail="visible conversation cleared",
                turn_index=self.session_state.turn_index,
            )
            return

        if command == "reset":
            self.agent.reset_session()
            return

        if command == "compact":
            self.agent.force_compact()
            return

        if command == "tools":
            tools = self.agent.tool_manager.list_tools()
            tool_lines = [f"{tool['name']}: {tool['description']}" for tool in tools]
            self.session_state.set_notice("Tools listed.")
            self.session_state.clear_current_trace()
            self.session_state.add_trace_entry(
                phase="command",
                summary="/tools",
                detail="\n".join(tool_lines) if tool_lines else "No tools linked.",
                turn_index=self.session_state.turn_index,
            )
            return

        if command == "help":
            self.session_state.set_notice("Slash commands available.")
            self.session_state.clear_current_trace()
            self.session_state.add_trace_entry(
                phase="command",
                summary="/help",
                detail=(
                    "/memory <text> | /memory <title> :: <content> | "
                    "/memory list [semantic|episodic] [n] | "
                    "/memory delete [semantic|episodic] [selector] | "
                    "/reset | /compact | /tools | /clear | /quit"
                ),
                turn_index=self.session_state.turn_index,
            )
            return

        if command == "memory":
            if remainder and not remainder.lower().startswith(("list ", "delete ")):
                self._add_semantic_memory(remainder)
                return
            self._run_memory_command(remainder)
            return

        self.session_state.set_notice(f"Unknown command: /{command}")
        self.session_state.clear_current_trace()
        self.session_state.add_trace_entry(
            phase="command",
            summary=f"/{command}",
            detail="use /help for available commands",
            turn_index=self.session_state.turn_index,
        )

    def _run_memory_command(self, remainder: str) -> None:
        parts = remainder.split()
        if not parts:
            self.session_state.set_notice("Use /memory list or /memory delete.")
            self.session_state.clear_current_trace()
            self.session_state.add_trace_entry(
                phase="command",
                summary="/memory",
                detail="use /memory list or /memory delete",
                turn_index=self.session_state.turn_index,
            )
            return

        subcommand = parts[0].lower()
        if subcommand == "list":
            kind = parts[1] if len(parts) > 1 else "semantic"
            limit = None
            if len(parts) > 2:
                try:
                    limit = int(parts[2])
                except ValueError:
                    limit = None
            titles = self.agent.memory_manager.list_memory_titles(kind, limit)
            if titles:
                detail = "\n".join(f"{index}. {title}" for index, title in enumerate(titles, 1))
                notice = f"{kind.title()} memories listed."
            else:
                detail = f"No {kind} memories found."
                notice = detail
            self.session_state.set_notice(notice)
            self.session_state.clear_current_trace()
            self.session_state.add_trace_entry(
                phase="command",
                summary=f"/memory list {kind}",
                detail=detail,
                turn_index=self.session_state.turn_index,
            )
            return

        if subcommand == "delete":
            if len(parts) >= 4 and parts[1].lower() == "all":
                kind = parts[2]
                result = self.agent.memory_manager.delete_all_memory(kind)
                detail = str(result)
                self.session_state.set_notice(f"Deleted {kind} memories.")
            elif len(parts) >= 3:
                kind = parts[1]
                selector = " ".join(parts[2:])
                result = self.agent.memory_manager.delete_memory(kind, selector)
                detail = str(result)
                self.session_state.set_notice(
                    f"Deleted {kind} memory." if result.get("deleted") else "No memory deleted."
                )
            else:
                detail = "Usage: /memory delete [semantic|episodic] [selector] or /memory delete all [kind]"
                self.session_state.set_notice(detail)
            self.session_state.clear_current_trace()
            self.session_state.add_trace_entry(
                phase="command",
                summary="/memory delete",
                detail=detail,
                turn_index=self.session_state.turn_index,
            )
            return

        self.session_state.set_notice("Unsupported memory command.")
        self.session_state.clear_current_trace()
        self.session_state.add_trace_entry(
            phase="command",
            summary="/memory",
            detail="use /memory list or /memory delete",
            turn_index=self.session_state.turn_index,
        )

    def _add_semantic_memory(self, text: str) -> None:
        note = text.strip()
        if not note:
            self.session_state.set_notice("Use /memory <note> to store a semantic memory.")
            self.session_state.clear_current_trace()
            self.session_state.add_trace_entry(
                phase="command",
                summary="/memory",
                detail="use /memory <note> to store a semantic memory",
                turn_index=self.session_state.turn_index,
            )
            return

        if "::" in note:
            title_text, content = [part.strip() for part in note.split("::", 1)]
        else:
            content = note
            title_words = note.split()
            title_text = " ".join(title_words[:6]).strip()
            if len(title_words) > 6:
                title_text = f"{title_text}..."

        title = title_text or "Untitled"
        entry = self.agent.memory_manager.add_semantic_memory(title=title, content=content)
        self.session_state.recent_dialogue = self.agent.memory_manager.recent_dialogue()
        self.session_state.set_memory_snapshot(self.agent.memory_manager.snapshot())
        self.session_state.clear_current_trace()
        if entry is None:
            self.session_state.set_notice("Semantic memory already exists or could not be stored.")
            detail = f"duplicate or empty semantic memory: {title}"
        else:
            self.session_state.set_notice(f"Stored semantic memory: {entry.title}")
            detail = f"stored semantic memory: {entry.title}"
        self.session_state.add_trace_entry(
            phase="command",
            summary="/memory",
            detail=detail,
            turn_index=self.session_state.turn_index,
        )

    def _run_curses(self, stdscr) -> None:
        self._screen = stdscr
        curses.curs_set(1)
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)
        stdscr.timeout(120)
        self._init_colors()

        while not self._should_exit:
            self._render()
            try:
                ch = stdscr.get_wch()
            except curses.error:
                self._frame += 1
                continue

            if isinstance(ch, str):
                if ch in ("\n", "\r"):
                    prompt = self.buffer_text
                    self.buffer_text = ""
                    self.submit_prompt(prompt)
                    continue
                if ch in ("\x7f", "\b"):
                    self.buffer_text = self.buffer_text[:-1]
                    continue
                if ch == "\x03":
                    break
                self.buffer_text += ch
            elif ch == curses.KEY_BACKSPACE:
                self.buffer_text = self.buffer_text[:-1]
            elif ch == curses.KEY_RESIZE:
                self._render()

    def _init_colors(self) -> None:
        if not curses.has_colors():
            return
        curses.start_color()
        curses.use_default_colors()
        palette = self._build_palette()
        curses.init_pair(PAIR_HEADER, palette["header"], -1)
        curses.init_pair(PAIR_USER, palette["user"], -1)
        curses.init_pair(PAIR_ASSISTANT, palette["assistant"], -1)
        curses.init_pair(PAIR_THOUGHTS, palette["thoughts"], -1)
        curses.init_pair(PAIR_FOOTER, palette["footer"], -1)
        curses.init_pair(PAIR_MUTED, palette["muted"], -1)
        curses.init_pair(PAIR_PRESENCE, palette["presence"], -1)
        curses.init_pair(PAIR_TELEMETRY, palette["telemetry"], -1)
        self._colors_ready = True

    def _build_palette(self) -> dict[str, int]:
        if curses.COLORS >= 256:
            return {
                "header": 173,
                "user": 130,
                "assistant": 130,
                "thoughts": 130,
                "footer": 173,
                "muted": 130,
                "presence": 220,
                "telemetry": 242,
            }
        return {
            "header": curses.COLOR_YELLOW,
            "user": curses.COLOR_YELLOW,
            "assistant": curses.COLOR_YELLOW,
            "thoughts": curses.COLOR_YELLOW,
            "footer": curses.COLOR_YELLOW,
            "muted": curses.COLOR_YELLOW,
            "presence": curses.COLOR_YELLOW,
            "telemetry": curses.COLOR_YELLOW,
        }

    def _color(self, pair: int) -> int:
        if self._colors_ready:
            return curses.color_pair(pair)
        return 0

    def _panel_geometry(self, width: int, height: int) -> tuple[int, int]:
        available = max(width, 20)
        panel_width = min(max(available - 12, 60), 84)
        panel_width = min(panel_width, max(available - 4, 20))
        x_offset = 4
        return panel_width, x_offset

    def _render_lines(
        self,
        lines,
        start_row: int,
        width: int,
        start_col: int = 0,
        color_pair: int = 0,
        bold: bool = False,
    ) -> int:
        if self._screen is None:
            return start_row

        row = start_row
        max_rows = self._screen.getmaxyx()[0]
        for line in lines:
            if row >= max_rows:
                break
            is_separator = line.startswith("─") and (" THOUGHTS " in line or " EXCHANGE " in line)
            pair = PAIR_HEADER if is_separator else color_pair
            attr = self._color(pair)
            if attr and (bold or is_separator):
                attr |= curses.A_BOLD
            try:
                self._screen.addnstr(row, start_col, line, width, attr)
            except curses.error:
                pass
            row += 1
        return row

    def _render(self) -> None:
        if self._screen is None:
            return

        stdscr = self._screen
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        panel_width, panel_x = self._panel_geometry(width, height)
        safe_width = max(panel_width, 20)
        safe_height = max(height, 10)
        section_width = max(min(safe_width, 78), 48)

        header_lines = build_header_lines(self.session_state, section_width)
        presence_lines = build_presence_lines(self.session_state, section_width, frame=self._frame)
        telemetry_line = build_telemetry_line(self.session_state, section_width)

        header_height = len(header_lines)
        presence_height = len(presence_lines)
        footer_height = 1
        section_gap = 2
        dialogue_footer_gap = 2
        # Keep a little breathing room above the input prompt so the last exchange
        # does not run right into the `> transmit thought...` line.
        content_height = max(
            safe_height - header_height - presence_height - footer_height - 3 - section_gap - dialogue_footer_gap,
            8,
        )
        thought_height = max(4, content_height // 2)
        dialogue_height = max(content_height - thought_height, 4)
        if thought_height != dialogue_height:
            thought_height = dialogue_height = max(content_height // 2, 4)
        thought_lines = build_thought_lines(self.session_state, section_width, thought_height)
        dialogue_lines = build_dialogue_lines(self.session_state, section_width, dialogue_height)

        row = 0
        row = self._render_lines(header_lines, row, section_width, start_col=panel_x, color_pair=PAIR_HEADER, bold=True)
        row += 1
        row = self._render_lines(presence_lines, row, section_width, start_col=panel_x, color_pair=PAIR_PRESENCE, bold=True)
        row += 1
        row = self._render_lines([telemetry_line], row, section_width, start_col=panel_x, color_pair=PAIR_TELEMETRY, bold=True)
        row += 1
        row = self._render_lines(thought_lines, row, section_width, start_col=panel_x, color_pair=PAIR_THOUGHTS)
        row += section_gap
        row = self._render_lines(dialogue_lines, row, section_width, start_col=panel_x, color_pair=PAIR_MUTED)

        footer_row = max(height - 1, 0)
        footer_lines = [build_input_line(self.buffer_text, section_width)]
        self._render_lines(footer_lines, footer_row, section_width, start_col=panel_x, color_pair=PAIR_FOOTER)

        stdscr.refresh()
        self._frame += 1
