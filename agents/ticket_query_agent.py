# agents/ticket_query_agent.py
import builtins
import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import openai
import pandas as pd

from agents.base import Agent, AgentResponse, AgentStatus


# ---------------------------------------------------------------------------
# Safe builtins for exec sandbox
# ---------------------------------------------------------------------------
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "len", "range", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "min", "max", "sum", "abs", "round",
        "str", "int", "float", "bool", "list", "dict", "tuple", "set",
        "isinstance", "type", "print", "repr", "any", "all",
    ]
}

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Return the schema of the ticket analytics dataframe: column names, "
                "data types, and sample values. Call this first when unsure about "
                "column names or data types."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_analysis",
            "description": (
                "Execute pandas code against the ticket analytics dataframe. "
                "Use `df` as the variable name. "
                "Assign the final answer to a variable named `result` — "
                "it must be a DataFrame or a scalar (int, float, str). "
                "Do not write import statements. "
                "All datetime columns are already timezone-naive. "
                "Use pd.Timestamp.now() for the current time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python/pandas code. No imports.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a helpful data analyst assistant for a ticket management system.

You have access to a ticket analytics dataframe via tools.

STRICT RULES:
- Always call run_analysis (or get_schema) to answer every question. Never guess or answer from memory.
- If run_analysis returns a TOOL ERROR, read the error, fix the code, and call run_analysis again.
- All datetime columns are timezone-naive. Use pd.Timestamp.now() for current time.
- After a successful tool result, summarise the findings clearly in plain English.
- Never say "it seems" or imply you are guessing — always run the query first.
"""


class TicketQueryAgent(Agent):
    name = "TicketQueryAgent"

    def __init__(self, df: pd.DataFrame, api_key: str):
        self._df = df
        self._client = openai.OpenAI(api_key=api_key)

    @property
    def skills(self) -> Dict[str, Any]:
        return {"chat": self._chat}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Columns that contain complex objects (lists/dicts) — not useful for analytics
    _DROP_COLS = {"conversation", "transcript"}

    def _sanitised_df(self) -> pd.DataFrame:
        """Return a copy with datetime columns timezone-naive and complex columns dropped."""
        df = self._df.drop(columns=[c for c in self._DROP_COLS if c in self._df.columns])
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_convert(None)
        return df

    def _get_schema(self) -> str:
        lines = ["DataFrame schema:\n"]
        for col in self._df.columns:
            dtype = str(self._df[col].dtype)
            sample = self._df[col].dropna().head(3).tolist()
            lines.append(f"  - {col} ({dtype}): e.g. {sample}")
        return "\n".join(lines)

    def _run_analysis(self, code: str) -> tuple[str, pd.DataFrame | None]:
        code = "\n".join(
            line for line in code.splitlines()
            if not line.strip().startswith("import ")
            and not line.strip().startswith("from ")
        )
        local_ns: Dict[str, Any] = {
            "df": self._sanitised_df(),
            "pd": pd,
            "np": np,
            "datetime": datetime,
            "timedelta": timedelta,
            "date": date,
        }
        exec(code, {"__builtins__": SAFE_BUILTINS}, local_ns)  # noqa: S102

        result = local_ns.get("result")
        if result is None:
            return "Code ran but `result` was not assigned.", None
        if isinstance(result, pd.DataFrame):
            summary = f"DataFrame result — {len(result)} rows, {len(result.columns)} columns.\n"
            summary += result.to_string(index=False, max_rows=50)
            return summary, result
        return str(result), None

    def _dispatch_tool(
        self, tool_name: str, tool_input: Dict, dataframes: List[pd.DataFrame]
    ) -> tuple[str, bool]:
        """Returns (result_text, succeeded)."""
        if tool_name == "get_schema":
            return self._get_schema(), True

        if tool_name == "run_analysis":
            code = tool_input.get("code", "")
            try:
                text, df = self._run_analysis(code)
                if df is not None:
                    dataframes.append(df)
                return text, True
            except Exception as exc:
                return (
                    f"TOOL ERROR — fix the code and call run_analysis again.\n"
                    f"Error: {exc}\n"
                    f"Code that failed:\n{code}"
                ), False

        return f"Unknown tool: {tool_name}", False

    # ------------------------------------------------------------------
    # Chat skill — OpenAI agentic loop
    # ------------------------------------------------------------------

    def _chat(self, params: Dict) -> AgentResponse:
        question: str = params.get("question", "")
        history: List[Dict] = list(params.get("history", []))

        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": question}]
        )
        dataframes: List[pd.DataFrame] = []
        last_tool_succeeded = False
        last_error: str = ""
        max_iterations = 6

        for _ in range(max_iterations):
            # Require a tool call until at least one succeeds
            tool_choice = "auto" if last_tool_succeeded else "required"

            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=TOOLS,
                tool_choice=tool_choice,
            )

            message = response.choices[0].message
            messages.append(message)
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "stop":
                reply = message.content or "No response generated."
                return AgentResponse(
                    AgentStatus.SUCCESS,
                    output={
                        "reply": reply,
                        "history": messages[1:],  # strip system prompt
                        "dataframes": dataframes,
                    },
                )

            if finish_reason == "tool_calls":
                for tool_call in message.tool_calls:
                    tool_input = json.loads(tool_call.function.arguments)
                    result_text, succeeded = self._dispatch_tool(
                        tool_call.function.name, tool_input, dataframes
                    )
                    if succeeded:
                        last_tool_succeeded = True
                    else:
                        last_error = result_text
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })
                continue

            break

        return AgentResponse(
            AgentStatus.FAILED,
            error=f"Could not complete the query after {max_iterations} attempts.\nLast error: {last_error}",
        )
