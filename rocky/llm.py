import os

try:
    from ollama import Client
except Exception:  # pragma: no cover - import guard for test environments
    Client = None

from rocky.conversation import PromptContext


class LLM:
    kind = "base"

    def __init__(
        self,
        model=None,
        client=None,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma4:e2b")
        if client is not None:
            self.client = client
        elif Client is not None:
            self.client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        else:
            self.client = _UnavailableOllamaClient()

    def build_system_prompt(
        self,
        template,
        tools_section,
        semantic_memory="",
        episodic_memory="",
        monologue="",
        emotion="neutral",
    ):
        return template.format(
            tools_section=tools_section,
            semantic_memory=semantic_memory,
            episodic_memory=episodic_memory,
            monologue=monologue,
            emotion=emotion,
        )

    @classmethod
    def build(cls, model=None, client=None):
        normalized = (model or "").lower()
        if "gemma4" in normalized:
            return Gemma4LLM(model=model, client=client)
        return ChatLLM(model=model, client=client)

    def generate_raw(self, context: PromptContext, think: bool = True):
        raise NotImplementedError

    def generate_stream(self, context: PromptContext):
        raise NotImplementedError

    @staticmethod
    def _dump_response(response):
        if isinstance(response, dict):
            return response
        return response.model_dump()


class Gemma4LLM(LLM):
    kind = "gemma"
    DEFAULT_MODEL = "gemma4:e2b"

    def __init__(
        self,
        model=None,
        client=None,
    ):
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            client=client,
        )

    def _format_turn(self, entry):
        role = entry.role
        content = entry.content

        if role == "system":
            return f"<|turn>system\n{content}<turn|>"
        if role == "user":
            return f"<|turn>user\n{content}<turn|>"
        if role == "assistant":
            return f"<|turn>model\n{content}<turn|>"
        if role == "tool":
            tool_name = entry.tool_name or "unknown_tool"
            return f"<|tool_response>response:{tool_name}{{result:<|\"|>{content}<|\"|>}}<tool_response|>"
        return f"<|turn>user\n{content}<turn|>"

    def build_prompt(self, context: PromptContext):
        prompt_parts = [f"<|turn>system\n{context.system_prompt}<turn|>"]
        prompt_parts.extend(self._format_turn(entry) for entry in context.dialogue)
        prompt_parts.append("<|turn>model\n")
        return "\n".join(prompt_parts)

    def generate_raw(self, context: PromptContext, think: bool = True):
        response = self.client.generate(
            model=self.model,
            prompt=self.build_prompt(context),
            raw=True,
            think=think,
        )
        raw = self._dump_response(response)
        return {
            "text": raw.get("response", ""),
            "reasoning": raw.get("thinking") or "",
            "raw": raw,
        }

    def generate_stream(self, context: PromptContext):
        response = self.client.generate(
            model=self.model,
            prompt=self.build_prompt(context),
            raw=True,
            think=True,
            stream=True,
        )
        for chunk in response:
            raw = self._dump_response(chunk)
            yield {
                "text_delta": raw.get("response", "") or "",
                "reasoning_delta": raw.get("thinking", "") or "",
                "done": bool(raw.get("done")),
                "raw": raw,
            }


class ChatLLM(LLM):
    kind = "chat"
    DEFAULT_MODEL = "llama3.1"

    def __init__(
        self,
        model=None,
        client=None,
    ):
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            client=client,
        )

    def _to_message(self, entry):
        role = entry.role
        content = entry.content

        if role == "tool":
            tool_name = entry.tool_name or "unknown_tool"
            return {
                "role": "system",
                "content": f"Tool result for {tool_name}: {content}",
            }
        if role == "assistant":
            return {"role": "assistant", "content": content}
        if role == "user":
            return {"role": "user", "content": content}
        return {"role": "system", "content": content}

    def build_messages(self, context: PromptContext):
        messages = [{"role": "system", "content": context.system_prompt}]
        messages.extend(self._to_message(entry) for entry in context.dialogue)
        return messages

    def generate_raw(self, context: PromptContext, think: bool = True):
        response = self.client.chat(
            model=self.model,
            messages=self.build_messages(context),
            think=think,
        )
        raw = self._dump_response(response)
        message = raw.get("message") or {}
        return {
            "text": message.get("content", "") or "",
            "reasoning": message.get("thinking") or "",
            "raw": raw,
        }

    def generate_stream(self, context: PromptContext):
        response = self.client.chat(
            model=self.model,
            messages=self.build_messages(context),
            think=True,
            stream=True,
        )
        for chunk in response:
            raw = self._dump_response(chunk)
            message = raw.get("message") or {}
            yield {
                "text_delta": message.get("content", "") or raw.get("response", "") or "",
                "reasoning_delta": message.get("thinking", "") or raw.get("thinking", "") or "",
                "done": bool(raw.get("done")),
                "raw": raw,
            }


class _UnavailableOllamaClient:
    def __getattr__(self, name):
        raise RuntimeError(
            "ollama is not installed; pass a client mock in tests or install the ollama package"
        )
