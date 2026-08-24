"""Agent module"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from pycodeloop.abc.confirm import Confirm
from pycodeloop.abc.provider import Provider, ProviderResponse, ToolCall, Usage
from pycodeloop.abc.tool import Tool
from pycodeloop.core.context_window import context_window_for
from pycodeloop.core.session import Message, Session
from pycodeloop.tools import DEFAULT_TOOLS

_COMPACT_KEEP_RECENT_TURNS = 2
_COMPACT_SUMMARY_PROMPT = (
    "Summarize the conversation so far in a concise paragraph, preserving "
    "concrete facts, decisions, file paths, and any unfinished work — this "
    "summary replaces the full history, so don't drop anything needed to "
    "continue the task."
)

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0

_MAX_EMPTY_RESPONSE_RETRIES = 2

_TOOL_RESULT_SUMMARIZE_THRESHOLD = 8_000
_TOOL_SUMMARY_PROMPT = (
    "Summarize this tool output concisely, preserving specific facts, "
    "file paths, error messages, and anything needed to continue the "
    "task — this replaces the full output."
)


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
        return True
    return isinstance(exc, TimeoutError | ConnectionError)


DEFAULT_SYSTEM_PROMPT = (
    "You are CodeLoop, an autonomous coding agent. You have tools to read, "
    "write, edit, and delete files, list directories, search the codebase "
    "(grep/glob), fetch web pages, and run shell commands. Use them to "
    "accomplish the user's request directly instead of only describing "
    "what to do. Investigate before acting: read relevant files and search "
    "for existing patterns rather than guessing. Prefer the smallest change "
    "that solves the request. Be concise — state what you did, not what "
    "you're about to do. write_file's content and edit_file's old_string/"
    "new_string are literal file text, never diff syntax — no '@@ ... @@' "
    "hunk headers, no leading '-'/'+' line markers."
)


class Agent:
    """Drives a provider through a tool-use loop until it stops."""

    def __init__(
        self,
        provider: Provider,
        fallback_providers: list[Provider] | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 25,
        max_history_turns: int | None = None,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, str, bool], None] | None = None,
        on_text_delta: Callable[[str], None] | None = None,
        confirm: Callable[[str, str], bool | str] | Confirm | None = None,
        on_usage: Callable[[Usage, Usage, float], None] | None = None,
        on_request: Callable[[int, int], None] | None = None,
        auto_compact: bool = True,
        compact_threshold: float = 0.8,
        on_context: Callable[[int, int], None] | None = None,
        on_compact_start: Callable[[], None] | None = None,
        on_compact_end: Callable[[int, int], None] | None = None,
        on_message: Callable[[], None] | None = None,
        on_retry: Callable[[int, float, Exception], None] | None = None,
        on_turn_end: Callable[[], None] | None = None,
        on_trace_event: Callable[[dict], None] | None = None,
        on_provider_fallback: (
            Callable[[Provider, Provider, Exception], None] | None
        ) = None,
    ) -> None:
        self.provider = provider
        self.fallback_providers = fallback_providers or []
        self._provider_chain = [provider, *self.fallback_providers]
        self._provider_index = 0
        self.on_provider_fallback = on_provider_fallback
        self.tools = {tool.name: tool for tool in (tools or DEFAULT_TOOLS)}
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_history_turns = max_history_turns
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.on_text_delta = on_text_delta
        self.confirm = confirm
        self.on_usage = on_usage
        self.on_request = on_request
        self.auto_compact = auto_compact
        self.compact_threshold = compact_threshold
        self.on_context = on_context
        self.on_compact_start = on_compact_start
        self.on_compact_end = on_compact_end
        self.on_message = on_message
        self.on_retry = on_retry
        self.on_turn_end = on_turn_end
        self.on_trace_event = on_trace_event
        self.usage = Usage()

    def _trace(self, event_type: str, **fields) -> None:
        if self.on_trace_event:
            with contextlib.suppress(Exception):
                self.on_trace_event({"type": event_type, **fields})

    def _safe_call(self, callback: Callable | None, *args) -> None:
        """Invokes a consumer-supplied `on_*` callback (UI rendering,
        storage persistence, etc.) without letting a bug on that side
        abort the turn/tool loop still in progress. Before this, an
        exception from e.g. `on_message` (a storage write failing) or
        `on_tool_result` (a rendering bug) propagated straight out of
        `Agent.run()`/`_run_tool_calls()`, killing the rest of the turn
        over what should have been a self-contained side effect."""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:
            self._trace(
                "callback_error",
                callback=getattr(callback, "__name__", repr(callback)),
                error=str(exc),
            )

    def _complete(self, **kwargs) -> ProviderResponse:
        """`provider.complete()` with retry + exponential backoff on
        transient failures (rate limits, 5xx, network blips) within a
        provider. If `fallback_providers` were given and the active
        provider still fails after its own retry budget (retryable or
        not), advances to the next provider in `_provider_chain` — a
        fixed list captured once at construction, walked by index so a
        provider already tried this run is never retried, and one
        earlier in the chain that recovers isn't skipped forever. A
        real error on the last provider in the chain still raises."""
        starting_index = self._provider_index
        last_exc: Exception | None = None

        for index in range(starting_index, len(self._provider_chain)):
            provider = self._provider_chain[index]
            delay = _RETRY_BASE_DELAY
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    response = provider.complete(**kwargs)
                    if index != starting_index and self.on_provider_fallback:
                        self.on_provider_fallback(
                            self.provider, provider, last_exc
                        )
                    self._provider_index = index
                    self.provider = provider
                    return response
                except Exception as exc:
                    last_exc = exc
                    if attempt >= _MAX_RETRIES or not _is_retryable(exc):
                        self._trace("provider_error", error=str(exc))
                        break
                    self._trace(
                        "retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(exc),
                    )
                    self._safe_call(self.on_retry, attempt + 1, delay, exc)
                    time.sleep(delay)
                    delay *= 2

        if last_exc is None:
            raise AssertionError(
                "_complete exited the provider loop without an exception "
                "— this is a bug"
            )
        raise last_exc

    def _notify_message(self) -> None:
        """Fired after every message is appended to the session — lets a
        caller persist incrementally instead of only after the whole
        (possibly long, multi-tool-call) turn finishes, so a crash
        mid-turn doesn't lose everything already done in it."""
        self._safe_call(self.on_message)

    def _tool_schemas(self) -> list[dict]:
        return [tool.schema() for tool in self.tools.values()]

    def _execute(
        self,
        name: str,
        arguments: dict,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, bool]:
        tool = self.tools.get(name)

        if tool is None:
            return f"Unknown tool: {name}", True

        if tool.dangerous:
            if self.confirm is None:
                return (
                    f"Refused to run dangerous tool '{name}' without a confirm "
                    "callback. Pass confirm=... or use --yes only from a trusted CLI.",
                    True,
                )
            try:
                preview = tool.preview(**arguments)
            except Exception as exc:
                return (
                    f"Tool '{name}' preview raised {exc.__class__.__name__}: {exc}",
                    True,
                )
            ask = (
                self.confirm.ask
                if isinstance(self.confirm, Confirm)
                else self.confirm
            )
            answer = ask(name, preview)
            if answer is not True:
                if isinstance(answer, str) and answer:
                    return f"User declined and said: {answer}", True
                return "User declined to run this tool.", True

        try:
            if tool.wants_cancel_event:
                result = tool.run(cancel_event=cancel_event, **arguments)
            else:
                result = tool.run(**arguments)
        except Exception as exc:
            return (
                f"Tool '{name}' raised {exc.__class__.__name__}: {exc}",
                True,
            )

        return result.output, result.is_error

    def _can_parallelize(self, calls: list[ToolCall]) -> bool:
        if len(calls) <= 1:
            return False

        names = [call.name for call in calls]
        for name in set(names):
            if names.count(name) <= 1:
                continue
            tool = self.tools.get(name)
            if tool is None or not tool.concurrent_safe:
                return False

        return not any(
            (tool := self.tools.get(call.name)) is not None and tool.dangerous
            for call in calls
        )

    def _run_tool_calls(
        self,
        calls: list[ToolCall],
        session: Session,
        cancel_event: threading.Event | None,
    ) -> None:
        for call in calls:
            self._trace("tool_call", name=call.name, arguments=call.arguments)
            self._safe_call(self.on_tool_call, call.name, call.arguments)

        if cancel_event and cancel_event.is_set():
            results = {call.id: ("Cancelled by user.", True) for call in calls}
        elif self._can_parallelize(calls):
            executor = ThreadPoolExecutor(max_workers=len(calls))
            futures = {
                call.id: executor.submit(
                    self._execute, call.name, call.arguments, cancel_event
                )
                for call in calls
            }
            results = {}
            for call in calls:
                tool = self.tools.get(call.name)
                timeout = tool.timeout if tool else None
                try:
                    results[call.id] = futures[call.id].result(timeout=timeout)
                except FutureTimeoutError:
                    results[call.id] = (
                        f"Tool '{call.name}' timed out after {timeout}s.",
                        True,
                    )
            executor.shutdown(wait=False)
        else:
            results = {}
            for call in calls:
                if cancel_event and cancel_event.is_set():
                    results[call.id] = ("Cancelled by user.", True)
                else:
                    results[call.id] = self._execute(
                        call.name, call.arguments, cancel_event
                    )

        for call in calls:
            result_text, is_error = results[call.id]
            if (
                not is_error
                and len(result_text) > _TOOL_RESULT_SUMMARIZE_THRESHOLD
            ):
                result_text = self._summarize_tool_result(
                    call.name, result_text
                )
            self._trace(
                "tool_result",
                name=call.name,
                is_error=is_error,
                result_len=len(result_text),
            )
            self._safe_call(
                self.on_tool_result, call.name, result_text, is_error
            )
            session.add_tool_result(call.id, result_text)
            self._notify_message()

    def _summarize_tool_result(self, tool_name: str, text: str) -> str:
        summary = self._complete(
            system_prompt="Summarize tool output concisely for context compaction.",
            messages=[
                Message(
                    role="user",
                    content=f"Tool '{tool_name}' returned:\n\n{text}\n\n{_TOOL_SUMMARY_PROMPT}",
                )
            ],
            tools=[],
        )
        self.usage = self.usage + summary.usage
        return f"[Summarized tool output — {len(text)} chars condensed]\n{summary.text}"

    def _compact(self, session: Session) -> None:
        """Summarize everything but the most recent turns via the
        provider itself, replacing the older history with one condensed
        message — keeps the conversation going instead of hitting the
        model's context limit."""
        history = session.history()
        turn_starts = [i for i, m in enumerate(history) if m.role == "user"]

        if len(turn_starts) <= _COMPACT_KEEP_RECENT_TURNS:
            return

        self._safe_call(self.on_compact_start)

        before_count = len(history)
        cutoff = turn_starts[-_COMPACT_KEEP_RECENT_TURNS]
        older, recent = history[:cutoff], history[cutoff:]

        summary = self._complete(
            system_prompt="Summarize conversations concisely for context compaction.",
            messages=[
                *older,
                Message(role="user", content=_COMPACT_SUMMARY_PROMPT),
            ],
            tools=[],
        )
        self.usage = self.usage + summary.usage

        recent[0] = Message(
            role=recent[0].role,
            content=(
                f"[Earlier conversation summary]\n{summary.text}\n\n---\n\n"
                f"{recent[0].content}"
            ),
            tool_call_id=recent[0].tool_call_id,
            tool_calls=recent[0].tool_calls,
            images=recent[0].images,
        )
        session.replace_messages(recent)
        self._notify_message()

        self._trace(
            "compact", before=before_count, after=len(session.messages)
        )
        self._safe_call(
            self.on_compact_end, before_count, len(session.messages)
        )

    def run(
        self,
        prompt: str,
        session: Session | None = None,
        images: list[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Runs until the model stops calling tools, `max_turns` is hit,
        or `cancel_event` is set — checked at each turn boundary and
        before every tool call, never mid-request, so an in-flight
        provider call always finishes first."""
        self._provider_index = 0
        self.provider = self._provider_chain[0]
        session = session or Session(system_prompt=self.system_prompt)
        session.add_user(prompt, images=images)
        self._notify_message()
        self._trace("run_start", prompt_len=len(prompt))

        if self.max_history_turns is not None:
            session.trim(self.max_history_turns)

        for _ in range(self.max_turns):
            if cancel_event and cancel_event.is_set():
                self._trace("run_end", reason="cancelled")
                return "Cancelled by user."

            context_window = getattr(self.provider, "context_window", None)
            if context_window is None:
                context_window = context_window_for(self.provider.model)
            if self.auto_compact and session.get_last_context_tokens() >= (
                context_window * self.compact_threshold
            ):
                self._compact(session)

            tools = self._tool_schemas()
            self._safe_call(
                self.on_request, len(session.history()), len(tools)
            )

            started_at = time.perf_counter()
            response = self._complete(
                system_prompt=self.system_prompt,
                messages=session.history(),
                tools=tools,
                on_delta=self.on_text_delta,
                cancel_event=cancel_event,
            )
            elapsed = time.perf_counter() - started_at

            if response.stop_reason == "cancelled":
                self.usage = self.usage + response.usage
                self._safe_call(
                    self.on_usage, response.usage, self.usage, elapsed
                )
                if response.text.strip():
                    session.add_assistant(response.text)
                    self._notify_message()
                    self._safe_call(self.on_turn_end)
                self._trace("run_end", reason="cancelled")
                return "Cancelled by user."

            empty_retries = 0
            while (
                not response.text.strip()
                and not response.tool_calls
                and empty_retries < _MAX_EMPTY_RESPONSE_RETRIES
            ):
                self.usage = self.usage + response.usage
                self._safe_call(
                    self.on_usage, response.usage, self.usage, elapsed
                )

                empty_retries += 1
                self._trace(
                    "empty_response_retry",
                    model=self.provider.model,
                    attempt=empty_retries,
                )
                self._safe_call(
                    self.on_retry,
                    empty_retries,
                    0.0,
                    RuntimeError(
                        f"{self.provider.model} returned an empty "
                        "response with no tool calls"
                    ),
                )
                started_at = time.perf_counter()
                response = self._complete(
                    system_prompt=self.system_prompt,
                    messages=session.history(),
                    tools=tools,
                    on_delta=self.on_text_delta,
                    cancel_event=cancel_event,
                )
                elapsed = time.perf_counter() - started_at

                if response.stop_reason == "cancelled":
                    self.usage = self.usage + response.usage
                    self._safe_call(
                        self.on_usage, response.usage, self.usage, elapsed
                    )
                    if response.text.strip():
                        session.add_assistant(response.text)
                        self._notify_message()
                        self._safe_call(self.on_turn_end)
                    self._trace("run_end", reason="cancelled")
                    return "Cancelled by user."

            if not response.text.strip() and not response.tool_calls:
                self.usage = self.usage + response.usage
                self._safe_call(
                    self.on_usage, response.usage, self.usage, elapsed
                )

                error_text = (
                    f"{self.provider.model} returned an empty response "
                    "with no tool calls after "
                    f"{_MAX_EMPTY_RESPONSE_RETRIES} retries. The model may "
                    "be too weak for agent mode with the current tool/"
                    "system-prompt size — try a larger model."
                )
                session.add_assistant(error_text)
                self._notify_message()
                self._safe_call(self.on_turn_end)
                self._trace("run_end", reason="empty_response")
                return error_text

            self.usage = self.usage + response.usage
            self._safe_call(self.on_usage, response.usage, self.usage, elapsed)

            self._trace(
                "turn",
                model=self.provider.model,
                tool_calls=[call.name for call in response.tool_calls],
                stop_reason=response.stop_reason,
                elapsed=elapsed,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            session.update_last_context_tokens(response.usage.input_tokens)
            self._safe_call(
                self.on_context, response.usage.input_tokens, context_window
            )

            tool_calls = [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    **({"extra": call.extra} if call.extra else {}),
                }
                for call in response.tool_calls
            ]
            session.add_assistant(response.text, tool_calls=tool_calls or None)
            self._notify_message()
            self._safe_call(self.on_turn_end)

            if not response.tool_calls:
                self._trace("run_end", reason="done")
                return response.text

            self._run_tool_calls(response.tool_calls, session, cancel_event)

            if cancel_event and cancel_event.is_set():
                self._trace("run_end", reason="cancelled")
                return "Cancelled by user."

        self._trace("run_end", reason="max_turns")
        return "Reached max_turns without finishing."
