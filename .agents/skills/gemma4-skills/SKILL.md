---
name: gemma4-skills
description: Use when formatting prompts for Gemma 4, including dialogue turns, multimodal placeholders, thinking mode, and tool-use tokens.
---

# Gemma 4 Skills

Use this skill when building or rewriting prompts for Gemma 4. Prefer the templates below when you need a ready-to-paste prompt structure.

## Core formatting

- Start turns with `<|turn>`.
- End turns with `<turn|>`.
- Use the role token immediately after `<|turn>`:
  - `system`
  - `user`
  - `model`

Example:

```text
<|turn>system
You are a helpful assistant.<turn|>
<|turn>user
Hello.<turn|>
```

## Prompt templates

### Text-only chat

```text
<|turn>system
You are a helpful assistant. Answer clearly and concisely.<turn|>
<|turn>user
{user_message}<turn|>
<|turn>model
```

### Multimodal prompt

```text
<|turn>system
You are a helpful assistant.<turn|>
<|turn>user
Describe this image: <|image|>

Also analyze this audio: <|audio|><turn|>
<|turn>model
```

### Thinking enabled

```text
<|turn>system
<|think|>
{system_instructions}<turn|>
<|turn>user
{user_message}<turn|>
<|turn>model
```

### Tool-use turn

```text
<|turn>system
<|think|>{system_instructions}<|tool>declaration:{tool_schema}<tool|><turn|>
<|turn>user
{user_message}<turn|>
<|turn>model
<|channel>thought
...
<channel|><|tool_call>call:{tool_name}{{{args}}}<tool_call|><|tool_response>
```

## Multimodal placeholders

- Use `<|image|>` where image embeddings should be inserted.
- Use `<|audio|>` where audio embeddings should be inserted.
- Keep these as placeholder tokens in the prompt text; the model replaces them after tokenization.

## Thinking mode

- Enable thinking by including `<|think|>` in the system turn.
- Keep thinking enabled at the conversation level, alongside other system instructions.
- If you need a no-thinking variant for later turns, remove the thinking token when you rebuild the prompt.

## Tool use

- Define tools with `<|tool>...<tool|>`.
- Request tool calls with `<|tool_call>...<tool_call|>`.
- Return tool results with `<|tool_response>...<tool_response|>`.
- Treat `<|tool_response>` as a stop sequence when parsing output.

## Structured strings

- Use `<|"|>` to delimit string values inside structured tool data.
- Wrap every string literal in function declarations, calls, and responses with that token.

## Thought handling

- Strip model-generated thoughts before sending a standard multi-turn conversation back to the model.
- Do not strip thoughts inside a single tool-calling turn.
- For long-running agents, summarize prior reasoning and feed back a compact summary instead of raw thoughts.

## Practical checks

- Keep system instructions, tool declarations, and thinking configuration in one system turn when possible.
- Prefer compact prompts that preserve role structure and special tokens exactly.
- Do not invent alternative delimiters for Gemma 4 control tokens.
- When in doubt, preserve the exact token order from the templates and only swap the placeholder text.
