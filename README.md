# Project Rocky Agent

![Rocky Agent TUI Demo](static/demo.png)

This project is created for **learning and experimenting with various AI agent concepts** including persona based AI agents, cognitive architectures, multi-layered memory systems, and localized reasoning loops.

## What Rocky Does

- Runs in a curses-based terminal UI.
- Maintains a lightweight internal state with selective reflection, emotion, and short monologue notes that can steer later replies.
- Uses separate memory paths for:
  - semantic memory: durable Markdown-backed notes
  - episodic memory: durable conversation summaries and excerpts learned from important turns
- Compacts long dialogue into a working-memory summary entry instead of repeatedly carrying the full raw transcript.
- Routes memory selectively so Rocky does not search memory on low-signal turns like acknowledgements or "no thanks".
- Supports tool calling through the configured tool registry.

## Getting Started

1. **Install dependencies**:

   ```bash
   poetry install
   ```

2. **Run the bootstrap setup**:
   This will automatically create your `.env`, initialize the database file (`rocky_memory.sqlite3`), and pull the required Ollama model:

   ```bash
   ./setup.sh
   ```

3. **Launch Rocky**:
   ```bash
   poetry run python rocky.py
   ```

4. **Launch voice mode**:
   ```bash
   poetry run python rocky.py voice
   ```

   Voice mode keeps Ollama/Gemma 4 as Rocky's reasoning model and uses separate local
   speech backends around it. First make sure the base Poetry environment exists:

   ```bash
   poetry install
   ```

   Then install the voice runtime packages. For Piper TTS:

   ```bash
   poetry run pip install faster-whisper sounddevice soundfile numpy piper-tts
   mkdir -p models/piper
   curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
     -o models/piper/en_US-lessac-medium.onnx
   curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
     -o models/piper/en_US-lessac-medium.onnx.json
   ```

   Useful environment variables:

   ```env
   ROCKY_VOICE_STT=faster-whisper
   ROCKY_VOICE_STT_MODEL=large-v3-turbo
   ROCKY_VOICE_TTS=piper
   ROCKY_PIPER_MODEL=models/piper/en_US-lessac-medium.onnx
   ROCKY_VOICE_LISTEN_MODE=continuous
   ROCKY_VOICE_RECORD_MODE=vad
   ROCKY_VOICE_RECORD_SECONDS=12
   ROCKY_VOICE_SILENCE_SECONDS=1.6
   ```

   If your `piper` executable is not on PATH, set it explicitly:

   ```env
   ROCKY_PIPER_EXECUTABLE=/absolute/path/to/piper
   ```

## Memory Model

### Semantic Memory

Semantic memory is stored as Markdown documents with:

- a title
- content

You can ingest one file or a directory of `.md` / `.markdown` files. Every top-level `# Heading` becomes a separate semantic memory document. Content between headings stays attached to that title.

Example:

```md
# Rocky's origin

Rocky comes from the Hail-Mary class and studies materials engineering.

# Rocky's expertise

Rocky prefers practical recommendations and tradeoff-based reasoning.
```

### Episodic Memory

Episodic memory stores learned conversation summaries and excerpts. It is used for prior-session and earlier-turn recall when the router decides the query needs it.

### Working Memory And Compaction

Rocky keeps recent dialogue in working memory. When the conversation gets too large, Rocky compacts older context into a system summary entry so the next turns still carry the important thread without replaying the whole transcript.

Important turns and compaction are intentionally separate:

- important turns use `learn()` to persist episodic and semantic memory
- compaction uses `summarize_dialogue()` to compress working context
- if both happen on the same turn, Rocky can reuse the learned episodic summary to avoid an extra model call

## TUI Commands

Inside the Rocky TUI, you can use the following slash commands:

- `/memory <text>`: Quick-store a semantic memory.
- `/memory <title> :: <content>`: Store a semantic memory with a specific title and content.
- `/memory search <title>`: Show the full stored semantic memory for a title.
- `/memory list [semantic|episodic] [n]`: List stored memories (defaults to semantic).
- `/memory delete [semantic|episodic] [selector]`: Delete a specific memory by index or title.
- `/memory delete all [semantic|episodic|all]`: Bulk delete memories.
- `/compact`: Manually force compaction of the current dialogue into working-memory summary form.
- `/tools`: List all tools currently available to Rocky.
- `/clear`: Clear the visible conversation history from the TUI (does not delete memory).
- `/reset`: Full session reset—wipes episodic memory and session snapshots while preserving semantic memory.
- `/voice`: Toggle continuous voice mode on or off. Enables hands-free conversation with STT/TTS running alongside the TUI.
- `/help`: Show available commands and their usage.
- `/quit` or `/exit`: Exit the Rocky session.

## Voice Mode

Voice mode is intentionally a layer around the existing text agent:

```text
microphone -> STT transcript -> RockyAgent -> TTS audio -> speaker
```

Voice mode can run push-to-talk or continuous listening. Push-to-talk waits for Enter before
each turn; continuous listening automatically starts each new turn after Rocky finishes speaking.

By default, voice mode now records until silence or until `ROCKY_VOICE_RECORD_SECONDS`
is reached. VAD keeps a short pre-roll and measures the minimum utterance length after
speech starts, so delayed starts and natural pauses are less likely to be clipped. Set
`ROCKY_VOICE_RECORD_MODE=fixed` to use the older fixed-duration recorder.

If Rocky cuts you off mid-thought, increase `ROCKY_VOICE_RECORD_SECONDS` or
`ROCKY_VOICE_SILENCE_SECONDS`. If Rocky misses quieter speech, lower
`ROCKY_VOICE_SILENCE_THRESHOLD`; if background noise keeps recording open, raise it.
`ROCKY_VOICE_PREROLL_SECONDS` controls how much audio before detected speech is kept.

Use `ROCKY_VOICE_LISTEN_MODE=continuous` for hands-free conversation. In continuous mode,
Rocky listens for speech, pauses while speaking the reply, then resumes listening. The default
`push_to_talk` mode waits for Enter before each turn.

## External CLI Commands

You can also manage Rocky's memory directly from your terminal without entering the TUI:

- **Ingest Markdown**: Import one or more files/directories as semantic memory:
  ```bash
  python rocky.py ingest datasets/semantic-memories.md
  ```
- **List Semantic Titles**:
  ```bash
  python rocky.py memory list [limit]
  ```
- **Search a Semantic Memory by Title**:
  ```bash
  python rocky.py memory search "User"
  ```

## Project Structure

```text
rocky/
├── rocky.py          # CLI entry point (Ingest / Memory management)
├── rocky/            # Core package
│   ├── agent.py      # Agent orchestration and turn execution logic
│   ├── config.py     # Configuration and environment loading
│   ├── conversation.py
│   ├── events.py     # Event-driven TUI updates
│   ├── llm.py        # Ollama-backed LLM interface
│   ├── memory/       # SQLite persistence and memory manager
│   ├── prompts/      # Externalized system and memory prompts
│   ├── session.py    # UI-facing session state management
│   ├── tools/        # Tool registration and execution
│   └── tui/          # Curses-based terminal interface
├── datasets/         # Example semantic memory Markdown files
└── pyproject.toml    # Dependencies and package metadata
```

## Notes

- **Memory Routing**: Rocky doesn't just search memory every turn; he uses a router to decide when past context is actually needed for the current query.
- **Compaction**: Dialogue is compacted into a working-memory summary when it grows too large, and you can trigger this manually with `/compact`.
- **Reflection**: Private reflection is gated so Rocky only pays for it on important turns or tool-driven turns.
- **Prompt Split**: The main reply prompt handles identity, memory, tools, and reply constraints; the monologue prompt carries Rocky's inner decision style and private thought framing.
- **Durable Storage**: Semantic memory (notes/facts) is persisted across all sessions, while episodic memory is unique to the current interaction timeline.

## Current Direction

- Keep reducing hidden model work in the hot path while preserving Rocky's personality.
- Improve semantic memory structure beyond title/content into richer entity- and relation-aware records.
- Continue tightening the boundary between outward reply generation and inward reflection/state updates.
