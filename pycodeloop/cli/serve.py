"""Command serve module — CodeLoop as a JSON-RPC-over-stdio server for
editor integrations (VSCode extension, etc). One JSON object per line
on stdout (notifications/responses) and stdin (requests); nothing else
may ever be written to stdout, or it corrupts the stream for the client."""

from __future__ import annotations

import json
import queue
import sys
import threading
import uuid

import typer

from pycodeloop.abc.provider import Usage
from pycodeloop.cli.flow import (
    PROVIDER_HELP,
    _load_mcp_tools,
    resolve_provider,
)
from pycodeloop.cli.render import console
from pycodeloop.core.codeloop import CodeLoop
from pycodeloop.core.config import Config
from pycodeloop.protocol.events import (
    CHAT_ALREADY_RUNNING,
    METHOD_NOT_FOUND,
    SERVER_ERROR,
    error_response,
    notification,
    response,
)

_HEARTBEAT_INTERVAL = 15.0
_CONFIRM_TIMEOUT = 120.0


class RpcServer:
    """Wires `Agent` callbacks to JSON-RPC notifications instead of the
    interactive CLI's `console.print` calls, and turns the synchronous
    `confirm()` gate into a request/response round-trip over the wire."""

    def __init__(
        self,
        flow: CodeLoop,
        provider_name: str,
        model_name: str,
        auto_approve: bool = False,
    ) -> None:
        self.flow = flow
        self.provider_name = provider_name
        self.model_name = model_name
        self.auto_approve = auto_approve
        self._out_lock = threading.Lock()
        self._confirm_waiters: dict[str, queue.Queue] = {}
        self._cancel_event: threading.Event | None = None
        self._chat_thread: threading.Thread | None = None
        self._disconnected = False
        self._wire_callbacks()

    def _send(self, message: dict) -> None:
        """Best-effort write of one NDJSON line to stdout. The client
        (editor extension) can disconnect mid-turn — closing its end of
        the pipe — at any point, including while a background thread is
        still streaming deltas for an in-flight turn. Once that happens
        every further write raises the same broken-pipe error, so this
        marks the server disconnected and gives up quietly instead of
        raising out of a callback (which would otherwise abort whatever
        turn/tool loop is in progress) or crashing a second time from
        inside an error handler that itself calls `_send`."""
        if self._disconnected:
            return
        line = json.dumps(message)
        try:
            with self._out_lock:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
        except OSError:
            self._disconnected = True

    def _notify(self, method: str, params: dict) -> None:
        self._send(notification(method, params))

    def _respond(self, request_id, result: dict) -> None:
        self._send(response(request_id, result))

    def _respond_error(self, request_id, code: int, message: str) -> None:
        self._send(error_response(request_id, code, message))

    def _wire_callbacks(self) -> None:
        agent = self.flow.agent

        def on_request(message_count: int, tool_count: int) -> None:
            self._notify(
                "chat/request",
                {"messageCount": message_count, "toolCount": tool_count},
            )

        def on_text_delta(delta: str) -> None:
            self._notify("chat/textDelta", {"delta": delta})

        def on_turn_end() -> None:
            self._notify("chat/turnEnd", {})

        def on_tool_call(name: str, args: dict) -> None:
            self._notify("chat/toolCall", {"name": name, "arguments": args})

        def on_tool_result(name: str, result: str, is_error: bool) -> None:
            self._notify(
                "chat/toolResult",
                {"name": name, "result": result, "isError": is_error},
            )

        def on_usage(turn: Usage, total: Usage, elapsed: float) -> None:
            self._notify(
                "chat/usage",
                {
                    "turnInputTokens": turn.input_tokens,
                    "turnOutputTokens": turn.output_tokens,
                    "totalInputTokens": total.input_tokens,
                    "totalOutputTokens": total.output_tokens,
                    "elapsed": elapsed,
                },
            )

        def on_context(used_tokens: int, limit_tokens: int) -> None:
            self._notify(
                "chat/context", {"used": used_tokens, "limit": limit_tokens}
            )

        def on_retry(attempt: int, delay: float, exc: Exception) -> None:
            self._notify(
                "chat/retry",
                {"attempt": attempt, "delay": delay, "error": str(exc)},
            )

        def on_compact_start() -> None:
            self._notify("chat/compactStart", {})

        def on_compact_end(before: int, after: int) -> None:
            self._notify("chat/compactEnd", {"before": before, "after": after})

        def confirm(name: str, preview: str) -> bool | str:
            if self.auto_approve:
                self._notify(
                    "chat/autoApproved", {"name": name, "preview": preview}
                )
                return True

            request_id = str(uuid.uuid4())
            answer_queue: queue.Queue = queue.Queue()
            self._confirm_waiters[request_id] = answer_queue
            self._notify(
                "chat/confirmRequest",
                {"id": request_id, "name": name, "preview": preview},
            )
            try:
                return answer_queue.get(timeout=_CONFIRM_TIMEOUT)
            except queue.Empty:
                self._notify("chat/confirmTimeout", {"id": request_id})
                return False
            finally:
                self._confirm_waiters.pop(request_id, None)

        agent.on_request = on_request
        agent.on_text_delta = on_text_delta
        agent.on_turn_end = on_turn_end
        agent.on_tool_call = on_tool_call
        agent.on_tool_result = on_tool_result
        agent.on_usage = on_usage
        agent.on_context = on_context
        agent.on_retry = on_retry
        agent.on_compact_start = on_compact_start
        agent.on_compact_end = on_compact_end
        agent.confirm = confirm

    def _run_heartbeat(self, stop: threading.Event) -> None:
        """Emits `chat/heartbeat` every `_HEARTBEAT_INTERVAL` seconds
        while a turn is in flight. Long reasoning or a long-running
        tool can otherwise leave the client with no message at all for
        minutes; a client with its own read timeout may then conclude
        the process died and drop the connection, losing the turn even
        though the server was still working on it."""
        while not stop.wait(_HEARTBEAT_INTERVAL):
            self._notify("chat/heartbeat", {})

    def _run_chat(self, request_id, params: dict) -> None:
        self._cancel_event = threading.Event()
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._run_heartbeat, args=(heartbeat_stop,), daemon=True
        )
        heartbeat_thread.start()
        try:
            result = self.flow.run(
                params.get("prompt", ""),
                session_key=params.get("sessionKey"),
                images=params.get("images"),
                cancel_event=self._cancel_event,
            )
            self._respond(request_id, {"text": result})
        except Exception as exc:
            self._respond_error(request_id, SERVER_ERROR, str(exc))
        finally:
            heartbeat_stop.set()

    def _run_ask(self, request_id, params: dict) -> None:
        try:
            text = self.flow.ask(params.get("prompt", ""))
            self._respond(request_id, {"text": text})
        except Exception as exc:
            self._respond_error(request_id, SERVER_ERROR, str(exc))

    def handle(self, request: dict) -> None:
        method = request.get("method")
        params = request.get("params") or {}
        request_id = request.get("id")

        if method == "chat/send":
            if self._chat_thread is not None and self._chat_thread.is_alive():
                self._respond_error(
                    request_id,
                    CHAT_ALREADY_RUNNING,
                    "A chat turn is already running.",
                )
            else:
                self._chat_thread = threading.Thread(
                    target=self._run_chat,
                    args=(request_id, params),
                    daemon=True,
                )
                self._chat_thread.start()
        elif method == "chat/ask":
            threading.Thread(
                target=self._run_ask, args=(request_id, params), daemon=True
            ).start()
        elif method == "chat/cancel":
            if self._cancel_event is not None:
                self._cancel_event.set()
            for waiter in list(self._confirm_waiters.values()):
                waiter.put(False)
        elif method == "chat/confirmResponse":
            waiter = self._confirm_waiters.get(params.get("id"))
            if waiter is not None:
                waiter.put(params.get("answer"))
        elif method == "session/list":
            storage = self.flow.config.storage
            sessions = (
                storage.list_sessions()
                if hasattr(storage, "list_sessions")
                else {}
            )
            self._respond(
                request_id,
                {
                    "sessions": [
                        {"key": key, **meta} for key, meta in sessions.items()
                    ]
                },
            )
        elif method == "session/load":
            storage = self.flow.config.storage
            key = params.get("key")
            session = storage.get(key) if storage is not None and key else None
            self._respond(
                request_id,
                {
                    "messages": [
                        {
                            "role": message.role,
                            "content": message.content,
                            "toolCallId": message.tool_call_id,
                            "toolCalls": message.tool_calls,
                            "images": message.images,
                        }
                        for message in (session.history() if session else [])
                    ]
                },
            )
        elif method == "initialize":
            self._respond(
                request_id,
                {"provider": self.provider_name, "model": self.model_name},
            )
        elif request_id is not None:
            self._respond_error(
                request_id, METHOD_NOT_FOUND, f"Unknown method: {method}"
            )

    def serve_forever(self) -> None:
        self._notify(
            "ready", {"provider": self.provider_name, "model": self.model_name}
        )
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                console.print(
                    f"[dim]⚠ dropped malformed request line ({exc}): "
                    f"{line[:200]!r}[/dim]"
                )
                continue
            self.handle(request)


def serve(
    provider: str = typer.Option(None, help=PROVIDER_HELP),
    model: str = typer.Option(None, help="Model name override."),
    base_url: str = typer.Option(
        None, help="Override the API endpoint (local/self-hosted servers)."
    ),
    url: str = typer.Option(
        None, help="Endpoint URL, required for --provider generic."
    ),
    mcp: list[str] = typer.Option(
        None, help="MCP server as 'command arg1 arg2'; repeatable."
    ),
    skills: bool = typer.Option(
        True,
        "--skills/--no-skills",
        help="Discover Claude/Cursor/AGENTS.md skills and expose a read_skill tool.",
    ),
    skills_refresh: bool = typer.Option(
        False, "--skills-refresh", help="Bypass the skills cache and rescan."
    ),
    delegate: bool = typer.Option(
        False,
        "--delegate/--no-delegate",
        help=(
            "Expose a delegate tool that spawns read-only sub-agents for "
            "independent subtasks, run in parallel. Off by default."
        ),
    ),
    memory: bool = typer.Option(
        True,
        "--memory/--no-memory",
        help=(
            "Load .pycodeloop/memory.md into the system prompt and expose "
            "a remember tool so standing corrections persist across "
            "sessions. On by default."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Auto-approve dangerous tool calls instead of round-tripping a confirmRequest.",
    ),
    workspace: bool = typer.Option(
        True,
        "--workspace/--no-workspace",
        help=(
            "Jail read_file/write_file/edit_file/delete_file/grep/glob to "
            "the working directory. Does NOT cover bash/git, which run "
            "arbitrary shell commands. On by default."
        ),
    ),
) -> None:
    """Run CodeLoop as a JSON-RPC-over-stdio server for editor
    integrations — no interactive terminal output, one JSON message
    per line on stdin/stdout."""
    console.file = sys.stderr

    provider_instance, provider_name = resolve_provider(
        provider, model, base_url, url
    )
    config = Config(
        provider=provider_instance,
        tools=_load_mcp_tools(mcp),
        skills=skills,
        skills_refresh=skills_refresh,
        delegation=delegate,
        memory=memory,
        workspace=workspace,
    )
    flow = CodeLoop(config=config)
    RpcServer(
        flow, provider_name, provider_instance.model, auto_approve=yes
    ).serve_forever()
