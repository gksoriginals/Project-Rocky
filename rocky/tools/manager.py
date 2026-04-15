import re

TOOL_CALL_BLOCK_RE = re.compile(r"<\|tool_call\>(.*?)<tool_call\|>", re.DOTALL)
THOUGHT_BLOCK_RE = re.compile(r"<\|channel\>thought.*?<channel\|>", re.DOTALL)
STRING_TOKEN = "<|\"|>"
META_PREFIX_RE = re.compile(
    r"^\s*(?:Internally:|Self-correction:|Reasoning:|speaking through the interface:)\s*",
    re.IGNORECASE,
)





class ToolManager:
    def __init__(self, tools_config):
        self.tools_config = tools_config
        self.tools = {name: config["function"] for name, config in tools_config.items()}

    def get_prompt_section(self, include_declarations=True):
        section = "Available tools:\n\n"
        declarations = []
        for name, config in self.tools_config.items():
            params = config.get("parameters", {})
            param_list = ", ".join([name for name in params.keys()])
            section += f"- {name}({param_list}): {config['description']}\n"
            for param_name, param_desc in params.items():
                section += f"  * {param_name}: {param_desc}\n"
            section += "\n"
            if include_declarations:
                signature = ", ".join(
                    f"{param_name}:{STRING_TOKEN}string{STRING_TOKEN}"
                    for param_name in params.keys()
                )
                declarations.append(f"<|tool>declaration:{name}{{{signature}}}<tool|>")

        if include_declarations and declarations:
            section += "Gemma 4 tool declarations:\n"
            section += "\n".join(declarations)
            section += "\n"
        return section

    def list_tools(self):
        return [
            {
                "name": name,
                "description": config["description"],
                "parameters": dict(config.get("parameters", {})),
            }
            for name, config in self.tools_config.items()
        ]

    def tool_count(self):
        return len(self.tools_config)

    def get_tool_metadata(self):
        return {
            "count": self.tool_count(),
            "names": [tool["name"] for tool in self.list_tools()],
        }

    def extract_tool_call(self, text):
        match = TOOL_CALL_BLOCK_RE.search(text)
        if match:
            parsed = self._parse_tool_call_payload(match.group(1))
            if parsed:
                return parsed

        try:
            import json

            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                import json

                return json.loads(match.group())
            except Exception:
                return None
        return None

    def _parse_tool_call_payload(self, payload):
        payload = payload.strip()
        if payload.startswith("call:"):
            payload = payload[len("call:"):]

        if "{" not in payload or not payload.endswith("}"):
            return None

        tool_name, raw_args = payload.split("{", 1)
        tool_name = tool_name.strip()
        raw_args = raw_args[:-1].strip()

        if not tool_name:
            return None

        args = {}
        if raw_args:
            pattern = re.compile(
                rf"\s*([A-Za-z_][\w-]*)\s*:\s*(?:{re.escape(STRING_TOKEN)}(.*?){re.escape(STRING_TOKEN)}|([^,{{}}]+))\s*(?:,|$)",
                re.DOTALL,
            )
            pos = 0
            while pos < len(raw_args):
                match = pattern.match(raw_args, pos)
                if not match:
                    return None
                key = match.group(1)
                value = match.group(2) if match.group(2) is not None else match.group(3)
                args[key] = value.strip()
                pos = match.end()

        return {"tool": tool_name, "args": args}

    def format_tool_response(self, tool_name, result):
        return (
            f"<|tool_response>response:{tool_name}"
            f"{{result:{STRING_TOKEN}{result}{STRING_TOKEN}}}"
            f"<tool_response|>"
        )

    def strip_thoughts(self, text):
        cleaned = THOUGHT_BLOCK_RE.sub("", text).strip()
        cleaned = META_PREFIX_RE.sub("", cleaned)
        cleaned = re.sub(r"(?im)^\s*(?:Internally:|Self-correction:|Reasoning:|speaking through the interface:)\s*", "", cleaned)
        return cleaned.strip()

    def execute(self, tool_call):
        result, _trace = self.execute_with_trace(tool_call)
        return result

    def execute_with_trace(self, tool_call):
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})

        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}", {
                "name": tool_name,
                "args": args,
                "error": f"Unknown tool: {tool_name}",
            }

        try:
            result = self.tools[tool_name](**args)
            return result, {
                "name": tool_name,
                "args": args,
                "result": result,
            }
        except Exception as e:
            return f"Tool error: {e}", {
                "name": tool_name,
                "args": args,
                "error": str(e),
            }
