"""Command chat module"""

from __future__ import annotations

import typer

from pycodeloop.cli.chat import CodeLoopApp
from pycodeloop.cli.flow import PROVIDER_HELP, build_flow


def chat(
    provider: str = typer.Option(None, help=PROVIDER_HELP),
    model: str = typer.Option(None, help="Model name override."),
    base_url: str = typer.Option(
        None, help="Override the API endpoint (local/self-hosted servers)."
    ),
    url: str = typer.Option(
        None, help="Endpoint URL, required for --provider generic."
    ),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature.",
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Maximum number of tokens to generate.",
    ),
    mcp: list[str] = typer.Option(
        None, help="MCP server as 'command arg1 arg2'; repeatable."
    ),
    skills: bool = typer.Option(
        True,
        "--skills/--no-skills",
        help=(
            "Discover Claude/Cursor/AGENTS.md skills and expose a "
            "read_skill tool. On by default."
        ),
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
        help="Skip confirmation prompts for dangerous tools.",
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
    """Start the full-screen interactive chat — real live redraw, no
    readline quirks, since Textual owns the whole terminal."""
    flow, provider_name, model_name = build_flow(
        provider,
        model,
        base_url,
        url,
        mcp,
        temperature=temperature,
        max_tokens=max_tokens,
        auto_approve=yes,
        skills=skills,
        skills_refresh=skills_refresh,
        delegation=delegate,
        memory=memory,
        workspace=workspace,
    )
    CodeLoopApp(flow, provider_name, model_name, auto_approve=yes).run()
