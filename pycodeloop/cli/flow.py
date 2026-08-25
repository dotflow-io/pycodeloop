"""Flow module"""

from __future__ import annotations

import queue
import shlex
import threading
import uuid
from pathlib import Path

import typer
from rich.panel import Panel

from pycodeloop.abc.provider import Provider, Usage
from pycodeloop.cli.render import (
    TurnBuffer,
    console,
    format_tokens,
    format_tool_call,
    render_preview,
    styled_text,
    tool_icon,
)
from pycodeloop.core.codeloop import CodeLoop
from pycodeloop.core.config import Config
from pycodeloop.mcp import MCPServer, MCPServerRegistry, load_mcp_tools
from pycodeloop.providers import BUNDLED_TEMPLATES, get_provider
from pycodeloop.settings import Settings
from pycodeloop.store.json_store import default_store
from pycodeloop.tools import DEFAULT_TOOLS

PROVIDER_HELP = (
    "path/to/config.json (see templates/) | generic (with --url) | "
    "'module.path:ClassName' for a custom Provider"
)

_CONFIRM_TIMEOUT = 3.0
_EOF = object()
_SESSION_IDS_SECTION = "session_ids"


def default_session_key() -> str:
    """A stable UUID per working directory, so `pycodeloop` resumes the
    same saved session when run again from the same project — the
    mapping (cwd -> id) lives in `~/.pycodeloop/config.json`, and the
    session's own `cwd` column is what stays human-readable."""
    cwd = str(Path.cwd())
    mapping = default_store.get_section(_SESSION_IDS_SECTION)

    session_id = mapping.get(cwd)
    if session_id is None:
        session_id = str(uuid.uuid4())
        mapping[cwd] = session_id
        default_store.set_section(_SESSION_IDS_SECTION, mapping)

    return session_id


def new_session_key() -> str:
    """Mint a fresh UUID for the current working directory, replacing
    whatever `default_session_key()` would otherwise resume — the old
    session stays in storage under its old key, just no longer linked
    to this cwd."""
    cwd = str(Path.cwd())
    mapping = default_store.get_section(_SESSION_IDS_SECTION)

    session_id = str(uuid.uuid4())
    mapping[cwd] = session_id
    default_store.set_section(_SESSION_IDS_SECTION, mapping)

    return session_id


def _load_mcp_tools(specs: list[str] | None) -> list:
    tools = list(DEFAULT_TOOLS)

    for spec in specs or []:
        if spec.startswith("saved:"):
            name = spec[len("saved:") :]
            server = MCPServerRegistry().load(name)

            if server is None:
                console.print(
                    f"{Settings.ERROR_ALERT} no saved MCP server named '{name}'"
                )
                raise typer.Exit(code=1)
        else:
            parts = shlex.split(spec)
            server = MCPServer(command=parts[0], args=parts[1:])

        console.print(
            f"{Settings.INFO_ALERT} connecting to MCP server '{spec}'"
        )
        tools.extend(load_mcp_tools(server))

    return tools


def resolve_provider(
    provider_name: str | None,
    model: str | None,
    base_url: str | None = None,
    url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[Provider, str]:
    """Build a `Provider` from CLI-style args, returning it alongside the
    resolved provider name — shared by the interactive CLI and `serve`."""
    provider_name = provider_name or Settings.PROVIDER
    is_json_config = (
        provider_name.endswith(".json") or provider_name in BUNDLED_TEMPLATES
    )

    if is_json_config:
        provider_kwargs: dict = {}
        if model:
            provider_kwargs["model"] = model
    elif provider_name == "generic":
        if not url:
            console.print(
                f"{Settings.ERROR_ALERT} --provider generic requires --url"
            )
            raise typer.Exit(code=1)
        provider_kwargs = {"model": model or Settings.MODEL, "url": url}
        if base_url:
            provider_kwargs["base_url"] = base_url
    else:
        provider_kwargs = {"model": model or Settings.MODEL}

    inference_params = {}

    if temperature is not None:
        inference_params["temperature"] = temperature

    if max_tokens is not None:
        inference_params["max_tokens"] = max_tokens

    if inference_params:
        provider_kwargs["inference_params"] = inference_params

    return get_provider(provider_name, **provider_kwargs), provider_name


def build_flow(
    provider_name: str | None,
    model: str | None,
    base_url: str | None = None,
    url: str | None = None,
    mcp: list[str] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    auto_approve: bool = False,
    skills: bool = False,
    skills_refresh: bool = False,
    delegation: bool = False,
    memory: bool = True,
    workspace: bool = True,
) -> tuple[CodeLoop, str, str]:
    provider, provider_name = resolve_provider(
        provider_name, model, base_url, url, temperature, max_tokens
    )

    buffer = TurnBuffer(console)

    def on_request(message_count: int, tool_count: int) -> None:
        console.print(
            f"[dim]💭 {provider_name}/{provider.model} — {message_count} msg in "
            f"context, {tool_count} tools available…[/dim]"
        )

    def on_tool_call(name: str, args: dict) -> None:
        buffer.flush()
        console.print(f"\n{format_tool_call(name, args)}")

    def on_tool_result(name: str, result: str, is_error: bool) -> None:
        preview = result if len(result) < 500 else result[:500] + "…"
        if result == "User declined to run this tool.":
            console.print("  [dim]⊘ skipped[/dim]")
        elif result.startswith("User declined and said: "):
            said = result[len("User declined and said: ") :]
            console.print(
                styled_text("  [bold white]↪ redirected:[/bold white] ", said)
            )
        elif is_error:
            console.print(
                styled_text("  [bold white]✗[/bold white] ", preview, "dim")
            )
        else:
            console.print(
                styled_text("  [bold white]✓[/bold white] ", preview, "dim")
            )

    def on_text_delta(delta: str) -> None:
        buffer.delta(delta)

    def on_usage(_turn: Usage, total: Usage, elapsed: float) -> None:
        buffer.flush()
        tokens = (
            format_tokens(total.input_tokens)
            + " in / "
            + format_tokens(total.output_tokens)
            + " out"
        )
        console.print(f"[dim]{Settings.ICON} {tokens} · {elapsed:.1f}s[/dim]")

    def on_context(used_tokens: int, limit_tokens: int) -> None:
        pct = round(100 * used_tokens / limit_tokens) if limit_tokens else 0
        console.print(f"[dim]{pct}% context used[/dim]")

    def on_compact_start() -> None:
        console.print("[dim]🗜 compacting context…[/dim]")

    def on_compact_end(before: int, after: int) -> None:
        console.print(
            f"[dim]✓ compacted context — {before} → {after} messages[/dim]"
        )

    def on_retry(attempt: int, delay: float, exc: Exception) -> None:
        console.print(
            styled_text(
                f"[dim]⚠ retrying ({attempt}/3) in {delay:.0f}s — ",
                str(exc),
                "dim",
            )
        )

    answer_queue: queue.Queue = queue.Queue()
    reader_started = False

    def ensure_reader() -> None:
        """One persistent stdin reader for the whole `run` invocation —
        a fresh thread per confirm() call would leave orphaned threads
        racing each other for stdin after every timeout."""
        nonlocal reader_started
        if reader_started:
            return
        reader_started = True

        def read_loop() -> None:
            while True:
                try:
                    answer_queue.put(input())
                except (EOFError, KeyboardInterrupt):
                    answer_queue.put(_EOF)
                    return
                except OSError:
                    return

        threading.Thread(target=read_loop, daemon=True).start()

    def confirm(name: str, preview: str) -> bool | str:
        if auto_approve:
            console.print("  [dim]running…[/dim]")
            return True

        console.print(
            Panel(
                render_preview(preview),
                title=f"{tool_icon(name)} {name}",
                border_style="grey50",
            )
        )
        console.print(
            f"{Settings.QUESTION_ALERT} run '{name}'? [y/n] (y): ", end=""
        )

        ensure_reader()
        try:
            answer = answer_queue.get(timeout=_CONFIRM_TIMEOUT)
        except queue.Empty:
            console.print(
                "\n  [dim]⏱ no response — running automatically[/dim]"
            )
            return True

        if answer is _EOF:
            console.print("\n  [dim]⊘ stdin closed — declining[/dim]")
            return False

        text = answer.strip()
        lowered = text.lower()
        if lowered in {"", "y", "yes"}:
            console.print("  [dim]running…[/dim]")
            return True
        if lowered in {"n", "no"}:
            return False
        return text

    config = Config(
        provider=provider,
        tools=_load_mcp_tools(mcp),
        skills=skills,
        skills_refresh=skills_refresh,
        delegation=delegation,
        memory=memory,
        workspace=workspace,
    )
    if config.skills:
        console.print(
            f"{Settings.INFO_ALERT} loaded {len(config.skills)} skill(s)"
        )

    flow = CodeLoop(config=config)
    flow.agent.on_request = on_request
    flow.agent.on_tool_call = on_tool_call
    flow.agent.on_tool_result = on_tool_result
    flow.agent.on_text_delta = on_text_delta
    flow.agent.confirm = confirm
    flow.agent.on_usage = on_usage
    flow.agent.on_context = on_context
    flow.agent.on_compact_start = on_compact_start
    flow.agent.on_compact_end = on_compact_end
    flow.agent.on_retry = on_retry
    return flow, provider_name, provider.model
