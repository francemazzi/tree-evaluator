from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Literal

from langchain_core.messages import AIMessage, BaseMessage


@dataclass(frozen=True)
class ToolLoopDecision:
    action: Literal["continue", "replan", "stop"]
    reason: str
    details: Optional[Dict[str, Any]] = None
    user_message: Optional[str] = None


class ToolLoopGuard:
    """Detects repeated tool calls and stops the graph to avoid infinite loops."""

    def __init__(self, max_consecutive_repeats: int = 2) -> None:
        self._max_consecutive_repeats = max_consecutive_repeats

    def evaluate(
        self,
        messages: Sequence[BaseMessage],
        last_fingerprint: Optional[str],
        repeat_count: int,
    ) -> Tuple[ToolLoopDecision, Optional[str], int]:
        fingerprint, details = self._extract_latest_tool_fingerprint(messages)
        if not fingerprint:
            return ToolLoopDecision("continue", "no_tool_call"), last_fingerprint, repeat_count

        # If we already have a conclusive tool result, answer deterministically immediately.
        conclusive = self._try_build_conclusive_answer(messages)
        if conclusive:
            return ToolLoopDecision("stop", "conclusive_tool_result", None, conclusive), fingerprint, 1

        if fingerprint == last_fingerprint:
            repeat_count += 1
        else:
            repeat_count = 1

        if repeat_count < self._max_consecutive_repeats:
            return ToolLoopDecision("continue", "below_threshold"), fingerprint, repeat_count

        # Replan: we are repeating the exact same tool call. Let the graph try to recover first.
        user_msg = self._build_user_message(details, repeat_count)
        return ToolLoopDecision("replan", "repeated_tool_call", details, user_msg), fingerprint, repeat_count

    def _extract_latest_tool_fingerprint(self, messages: Sequence[BaseMessage]) -> Tuple[Optional[str], Dict[str, Any]]:
        # Find the most recent AIMessage that contains tool_calls
        last_tool_ai: Optional[AIMessage] = None
        for msg in reversed(list(messages)):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                last_tool_ai = msg
                break

        if not last_tool_ai:
            return None, {}

        tool_calls = getattr(last_tool_ai, "tool_calls", None) or []
        normalized = []
        pretty = []
        for tc in tool_calls:
            try:
                name = tc.get("name") if isinstance(tc, dict) else None
                args = tc.get("args") if isinstance(tc, dict) else None
            except Exception:
                name, args = None, None
            if not name:
                continue
            args_str = self._stable_serialize(args)
            normalized.append(f"{name}:{args_str}")
            pretty.append({"name": name, "args": args})

        if not normalized:
            return None, {}

        # If multiple tool calls happen in a single step, treat them as a set (order-insensitive)
        normalized_sorted = "|".join(sorted(normalized))

        # Best-effort extract SQL from the latest tool result message content (if available)
        sql = self._try_extract_latest_sql(messages)
        details: Dict[str, Any] = {"tool_calls": pretty, "sql": sql, "fingerprint": normalized_sorted}
        return normalized_sorted, details

    def _stable_serialize(self, value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(value)

    def _try_extract_latest_sql(self, messages: Sequence[BaseMessage]) -> Optional[str]:
        # ToolNode usually adds ToolMessage objects; their type varies, so we parse by content heuristics.
        for msg in reversed(list(messages)):
            content = getattr(msg, "content", None)
            if not isinstance(content, str) or "sql_executed" not in content:
                continue
            # content may be a Python dict string with single quotes; try literal_eval.
            try:
                parsed = ast.literal_eval(content)
                if isinstance(parsed, dict):
                    sql = parsed.get("sql_executed") or parsed.get("sql_attempted")
                    if isinstance(sql, str) and sql.strip():
                        return sql.strip()
            except Exception:
                continue
        return None

    def _try_extract_latest_tool_result(self, messages: Sequence[BaseMessage]) -> Optional[Dict[str, Any]]:
        """
        Tool messages often contain JSON string results (e.g. DatasetQueryTool returns dict with 'results').
        We parse the most recent JSON object we can.
        """
        for msg in reversed(list(messages)):
            content = getattr(msg, "content", None)
            if not isinstance(content, str):
                continue
            content_str = content.strip()
            if not content_str:
                continue
            # Prefer JSON first (tool outputs in this project are JSON-serializable dicts)
            try:
                parsed = json.loads(content_str)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            # Fallback: some tool messages may be Python dict string
            try:
                parsed2 = ast.literal_eval(content_str)
                if isinstance(parsed2, dict):
                    return parsed2
            except Exception:
                continue
        return None

    def _try_build_conclusive_answer(self, messages: Sequence[BaseMessage]) -> Optional[str]:
        """
        If the last tool result already contains a definitive answer (e.g. top-1 aggregation),
        return a final user-facing answer to avoid extra LLM turns.
        """
        result = self._try_extract_latest_tool_result(messages)
        if not isinstance(result, dict):
            return None

        rows = result.get("results")
        if not isinstance(rows, list) or not rows:
            return None

        # Handle common "top-1" aggregation row: {genus_species: ..., count: ...}
        row0 = rows[0]
        if isinstance(row0, dict) and "genus_species" in row0 and "count" in row0:
            species = str(row0.get("genus_species") or "").strip()
            count_val = row0.get("count")
            count_int = None
            try:
                count_int = int(count_val)
            except Exception:
                pass

            if species and count_int is not None:
                count_it = self._format_int_it(count_int)
                return (
                    f"A Milano la specie più diffusa è {species}: {count_it} alberi\n\n"
                    f"Tool utilizzati: Dataset Query Tool"
                )

        # Handle single-value results (COUNT, etc.)
        if "result" in result and "column" in result:
            val = result.get("result")
            col = str(result.get("column") or "").strip()
            if col and isinstance(val, (int, float, str)):
                return f"{col}: {val}\n\nTool utilizzati: Dataset Query Tool"

        return None

    def _format_int_it(self, n: int) -> str:
        s = str(abs(int(n)))
        groups = []
        while len(s) > 3:
            groups.append(s[-3:])
            s = s[:-3]
        groups.append(s)
        grouped = ".".join(reversed(groups))
        return f"-{grouped}" if n < 0 else grouped

    def _build_user_message(self, details: Dict[str, Any], repeat_count: int) -> str:
        tool_calls = details.get("tool_calls") or []
        sql = details.get("sql")

        tool_name = tool_calls[0].get("name") if tool_calls else "tool"
        args = tool_calls[0].get("args") if tool_calls else None
        args_preview = ""
        if isinstance(args, dict):
            # show a short preview
            if "natural_query" in args:
                args_preview = f"Query: {args.get('natural_query')}"
            else:
                args_preview = self._stable_serialize({k: args[k] for k in list(args.keys())[:4]})

        sql_block = f"\n\nSQL ripetuta:\n{sql}" if sql else ""

        return (
            f"⚠️ Sembra che io stia ripetendo **la stessa chiamata tool** ({tool_name}) "
            f"per {repeat_count} volte senza fare progressi.\n\n"
            f"{args_preview}{sql_block}\n\n"
            "Per sbloccare la risposta, mi confermi cosa vuoi ottenere esattamente?\n"
            "- Vuoi **il risultato 1° in classifica** (top 1) oppure una **top 10**?\n"
            "- Vuoi considerare la colonna `genus_species` (specie completa) o separare `genere`/`specie`?\n\n"
            "Tool utilizzati: Dataset Query Tool"
        )


