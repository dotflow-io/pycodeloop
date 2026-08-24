"""Chat module"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Header, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from pycodeloop.abc.provider import Usage
from pycodeloop.cli.clipboard import read_clipboard_image_base64
from pycodeloop.cli.flow import default_session_key, new_session_key
from pycodeloop.cli.render import (
    format_tokens,
    format_tool_call,
    git_branch,
    render_preview,
    tool_icon,
)
from pycodeloop.core.codeloop import CodeLoop
from pycodeloop.core.session import Session

_COMMANDS = [
    ("/model", "show or switch the current model"),
    ("/reload", "reload the provider from its JSON config"),
    ("/new", "start a new session in this project"),
    ("/paste", "attach the clipboard's image to your next message"),
]

_THINKING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class PromptArea(TextArea):
    """Multi-line prompt box. Enter submits; shift+enter/alt+enter/ctrl+j
    inserts a newline — pasted multi-line text (e.g. a traceback) goes in
    untouched since paste bypasses key bindings entirely. shift+enter only
    works in terminals with the extended keyboard protocol (Kitty, iTerm2,
    WezTerm, Ghostty); alt+enter and ctrl+j work everywhere. A trailing
    backslash before Enter also inserts a newline (the backslash is
    consumed) — a universal fallback that needs no terminal support at
    all, for terminals (plain Terminal.app, tmux, VS Code's integrated
    terminal) that report every modifier combination as plain Enter.

    ctrl+v/cmd+v check the clipboard for an image before falling
    through to TextArea's own paste-as-text binding, which would
    otherwise consume the key first and paste-as-text always wins."""

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    async def _on_key(self, event) -> None:
        menu = self.app.query_one("#command-menu", OptionList)
        if menu.display:
            if event.key == "down":
                menu.action_cursor_down()
                event.prevent_default()
                event.stop()
                return
            if event.key == "up":
                menu.action_cursor_up()
                event.prevent_default()
                event.stop()
                return
            if event.key == "tab" and menu.highlighted_option is not None:
                self.text = f"{menu.highlighted_option.id} "
                self.move_cursor(self.document.end)
                self.app.hide_command_menu()
                event.prevent_default()
                event.stop()
                return
            if event.key == "enter" and menu.highlighted_option is not None:
                if self.text.strip() != menu.highlighted_option.id:
                    self.text = f"{menu.highlighted_option.id} "
                    self.move_cursor(self.document.end)
                    event.prevent_default()
                    event.stop()
                self.app.hide_command_menu()
            if event.key == "escape":
                self.app.hide_command_menu()
                event.prevent_default()
                event.stop()
                return

        if event.key == "escape":
            event.prevent_default()
            event.stop()
            self.app.cancel_turn()
            return
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            line, column = self.cursor_location
            before_cursor = self.document.get_line(line)[:column]
            if before_cursor.endswith("\\"):
                self.action_delete_left()
                self.insert("\n")
            else:
                self.post_message(self.Submitted(self.text))
            return
        if event.key in ("shift+enter", "alt+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if event.key in ("ctrl+v", "cmd+v"):
            image = await asyncio.to_thread(read_clipboard_image_base64)
            if image is not None:
                event.prevent_default()
                event.stop()
                self.app.attach_image(image)
                return
        await super()._on_key(event)


class CodeLoopApp(App):
    """Full-screen chat UI. Textual owns the whole terminal, so there is
    no second stdin reader to race — the class of bug that broke the
    plain-terminal chat's live redraw doesn't apply here."""

    CSS = """
    Screen {
        min-width: 20;
        min-height: 10;
    }
    #log {
        border: round grey;
        padding: 0 1;
        align-vertical: bottom;
    }
    #command-menu {
        display: none;
        dock: bottom;
        margin-bottom: 6;
        height: auto;
        max-height: 8;
        border: round grey;
    }
    PromptArea {
        border: round grey;
        dock: bottom;
        height: 6;
    }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit")]
    ENABLE_COMMAND_PALETTE = False

    def __init__(
        self,
        flow: CodeLoop,
        provider_name: str,
        model_name: str,
        auto_approve: bool = False,
    ) -> None:
        super().__init__(ansi_color=True)
        self.flow = flow
        self.provider_name = provider_name
        self.model_name = model_name
        self._auto_approve = auto_approve
        self.session_key = default_session_key()
        self._context_pct: int | None = None
        self._text_buffer = ""
        self._awaiting_confirm = False
        self._confirm_queue: queue.Queue = queue.Queue()
        self._stale_confirm_answer = False
        self._stale_expires_at = 0.0
        self._busy = False
        self._cancel_event: threading.Event | None = None
        self._pending: list[tuple[str, list[str] | None]] = []
        self._pending_images: list[str] = []
        self._thinking_static: Static | None = None
        self._thinking_timer = None
        self._thinking_meta = (0, 0)
        self._thinking_frame = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log")
        yield OptionList(id="command-menu")
        yield PromptArea(placeholder="Type a message…", id="prompt")

    def hide_command_menu(self) -> None:
        self.query_one("#command-menu", OptionList).display = False

    def _update_command_menu(self, text: str) -> None:
        menu = self.query_one("#command-menu", OptionList)
        if not text.startswith("/") or " " in text:
            menu.display = False
            return

        matches = [
            (name, desc) for name, desc in _COMMANDS if name.startswith(text)
        ]
        if not matches:
            menu.display = False
            return

        menu.clear_options()
        menu.add_options(
            Option(f"{name}  [dim]{desc}[/dim]", id=name)
            for name, desc in matches
        )
        menu.highlighted = 0
        menu.display = True

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "prompt":
            self._update_command_menu(event.text_area.text)

    def on_mount(self) -> None:
        self.title = "CodeLoop"
        self._update_subtitle()
        branch = git_branch()
        cwd_line = f"[dim]{os.getcwd()}[/dim]"
        if branch:
            cwd_line += f"[grey70]  ({branch})[/grey70]"
        self._log(
            Panel(
                f"[bold white]CodeLoop[/bold white] ready\n"
                f"[grey70]{self.provider_name}[/grey70] · "
                f"[grey70]{self.model_name}[/grey70]\n"
                f"{cwd_line}",
                border_style="grey50",
            )
        )
        self._restore_history()
        self.query_one("#prompt", PromptArea).focus()

        self.flow.agent.on_request = self._on_request
        self.flow.agent.on_tool_call = self._on_tool_call
        self.flow.agent.on_tool_result = self._on_tool_result
        self.flow.agent.on_text_delta = self._on_text_delta
        self.flow.agent.on_usage = self._on_usage
        self.flow.agent.on_context = self._on_context
        self.flow.agent.on_compact_start = self._on_compact_start
        self.flow.agent.on_compact_end = self._on_compact_end
        self.flow.agent.on_retry = self._on_retry
        self.flow.agent.confirm = self._confirm

    def _restore_history(self) -> None:
        """Render a previously saved session's user/assistant turns into
        the log on startup — the session itself is already resumed by
        `flow.run()`; this only makes that resume visible on screen."""
        storage = self.flow.config.storage
        if storage is None:
            return

        stored = storage.get(self.session_key)
        if stored is None or not stored.messages:
            return

        for message in stored.messages:
            if message.role == "user":
                self._log(
                    Panel(
                        Text(message.content),
                        border_style="bold white",
                        subtitle="[bold black on white] You [/bold black on white]",
                        subtitle_align="right",
                    )
                )
            elif message.role == "assistant" and message.content:
                self._log(
                    Panel(
                        Markdown(message.content),
                        border_style="grey50",
                        subtitle=(
                            "[bold white on grey30] Agent [/bold white on grey30]"
                        ),
                        subtitle_align="right",
                    )
                )

    def _log(self, renderable) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(Static(renderable))
        log.scroll_end(animate=False)

    @staticmethod
    def _styled(prefix_markup: str, plain: str, style: str = "") -> Text:
        """A self-contained `prefix_markup` (its own tags balanced)
        followed by `plain` appended as literal text, never parsed as
        markup — an arbitrary `[` in tool output/file content/user text
        can otherwise crash the whole app with a `MarkupError` deep
        inside Textual's compositor."""
        text = Text.from_markup(prefix_markup)
        text.append(plain, style=style or None)
        return text

    def _handle_command(self, text: str) -> bool:
        """Handle a `/model`/`/reload` slash command. Returns True if
        `text` was a command (caller should not send it to the agent)."""
        if text == "/model":
            self._log(
                self._styled(
                    "[dim]current model: ",
                    self.flow.agent.provider.model,
                    "dim",
                )
            )
            return True

        if text.startswith("/model "):
            self.flow.agent.provider.model = text[len("/model ") :].strip()
            self.model_name = self.flow.agent.provider.model
            self._update_subtitle()
            self._log(
                self._styled(
                    "[dim]switched to model: ", self.model_name, "dim"
                )
            )
            return True

        if text == "/new":
            self.session_key = new_session_key()
            self.flow.session = Session(
                system_prompt=self.flow.agent.system_prompt, cwd=os.getcwd()
            )
            self.query_one("#log", VerticalScroll).remove_children()
            self._log(
                "[dim]started a new session — previous history stays "
                "saved but won't resume here[/dim]"
            )
            return True

        if text == "/paste":
            self.run_worker(self._paste_from_command(), thread=True)
            return True

        if text == "/reload":
            reload = getattr(self.flow.agent.provider, "reload", None)
            if reload is None:
                self._log(
                    "[dim]this provider wasn't loaded from a JSON "
                    "config, nothing to reload[/dim]"
                )
                return True
            reload()
            self.model_name = self.flow.agent.provider.model
            self._update_subtitle()
            self._log(
                self._styled("[dim]reloaded — model: ", self.model_name, "dim")
            )
            return True

        return False

    def on_prompt_area_submitted(self, event: PromptArea.Submitted) -> None:
        prompt = self.query_one("#prompt", PromptArea)
        text = event.value.strip()
        prompt.text = ""

        if self._awaiting_confirm:
            self._confirm_queue.put(text)
            return

        if not text:
            return
        if text in {"exit", "quit"}:
            self.exit()
            return
        if self._handle_command(text):
            return

        images = self._pending_images or None
        self._pending_images = []

        label = "You" if not images else f"You ({len(images)} image(s))"
        self._log(
            Panel(
                Text(text),
                border_style="bold white",
                subtitle=f"[bold black on white] {label} [/bold black on white]",
                subtitle_align="right",
            )
        )

        if self._busy:
            self._pending.append((text, images))
            return

        self._start_turn(text, images)

    def attach_image(self, image: str) -> None:
        self._pending_images.append(image)
        self._log(
            f"[dim]📎 image attached ({len(self._pending_images)} pending)[/dim]"
        )

    async def _paste_from_command(self) -> None:
        image = await asyncio.to_thread(read_clipboard_image_base64)
        if image is None:
            self.call_from_thread(
                self._log, "[dim]⊘ no image on the clipboard[/dim]"
            )
            return
        self.call_from_thread(self.attach_image, image)

    def _start_turn(self, text: str, images: list[str] | None = None) -> None:
        self._busy = True
        self._cancel_event = threading.Event()
        self.run_worker(self._run_turn(text, images), thread=True)

    def cancel_turn(self) -> None:
        if not self._busy or self._cancel_event is None:
            return
        self._cancel_event.set()
        self._stop_thinking()
        self._log("[dim]⊘ cancelling…[/dim]")

    async def _run_turn(
        self, text: str, images: list[str] | None = None
    ) -> None:
        try:
            self.flow.run(
                text,
                session_key=self.session_key,
                images=images,
                cancel_event=self._cancel_event,
            )
        except Exception as exc:
            self.call_from_thread(self._stop_thinking)
            if self._text_buffer.strip():
                self.call_from_thread(
                    self._log,
                    Panel(
                        Markdown(self._text_buffer),
                        border_style="grey50",
                        title="[dim]interrupted[/dim]",
                        subtitle=(
                            "[bold white on grey30] Agent [/bold white on grey30]"
                        ),
                        subtitle_align="right",
                    ),
                )
                self._text_buffer = ""
            self.call_from_thread(
                self._log,
                self._styled("[bold white]✗ Error:[/bold white] ", str(exc)),
            )
        finally:
            self.call_from_thread(self._finish_turn)

    def _finish_turn(self) -> None:
        self._busy = False
        if self._cancel_event is not None and self._cancel_event.is_set():
            self._log("[dim]⊘ cancelled[/dim]")
        if self._pending:
            text, images = self._pending.pop(0)
            self._start_turn(text, images)

    def _on_request(self, message_count: int, tool_count: int) -> None:
        self.call_from_thread(self._start_thinking, message_count, tool_count)

    def _start_thinking(self, message_count: int, tool_count: int) -> None:
        self._thinking_frame = 0
        static = Static("")
        self.query_one("#log", VerticalScroll).mount(static)
        static.scroll_visible()
        self._thinking_static = static
        self._thinking_meta = (message_count, tool_count)
        self._tick_thinking()
        self._thinking_timer = self.set_interval(0.15, self._tick_thinking)

    def _tick_thinking(self) -> None:
        if self._thinking_static is None:
            return

        frame = _THINKING_FRAMES[self._thinking_frame % len(_THINKING_FRAMES)]
        self._thinking_frame += 1
        message_count, tool_count = self._thinking_meta
        pct = (
            f" · {self._context_pct}% context"
            if self._context_pct is not None
            else ""
        )
        self._thinking_static.update(
            f"[dim]{frame} thinking — {self.provider_name}/{self.model_name} · "
            f"{message_count} msg, {tool_count} tools{pct}…[/dim]"
        )

    def _stop_thinking(self) -> None:
        if self._thinking_timer is not None:
            self._thinking_timer.stop()
            self._thinking_timer = None
        if self._thinking_static is not None:
            message_count, tool_count = self._thinking_meta
            pct = (
                f" · {self._context_pct}% context"
                if self._context_pct is not None
                else ""
            )
            self._thinking_static.update(
                f"[dim]💭 {self.provider_name}/{self.model_name} — "
                f"{message_count} msg, {tool_count} tools{pct}[/dim]"
            )
        self._thinking_static = None

    def _on_context(self, used_tokens: int, limit_tokens: int) -> None:
        self._context_pct = (
            round(100 * used_tokens / limit_tokens) if limit_tokens else 0
        )
        self.call_from_thread(self._update_subtitle)

    def _on_compact_start(self) -> None:
        self.call_from_thread(self._log, "[dim]🗜 compacting context…[/dim]")

    def _on_compact_end(self, before: int, after: int) -> None:
        self.call_from_thread(
            self._log,
            f"[dim]✓ compacted context — {before} → {after} messages[/dim]",
        )

    def _on_retry(self, attempt: int, delay: float, exc: Exception) -> None:
        self.call_from_thread(self._start_retry_countdown, attempt, delay, exc)

    def _start_retry_countdown(
        self, attempt: int, delay: float, exc: Exception
    ) -> None:
        static = Static("")
        self.query_one("#log", VerticalScroll).mount(static)
        static.scroll_visible()
        deadline = time.monotonic() + delay
        error_text = str(exc)

        def tick() -> None:
            remaining = max(0.0, deadline - time.monotonic())
            static.update(
                self._styled(
                    f"[dim]⚠ retrying ({attempt}/3) in {remaining:.0f}s — ",
                    error_text,
                    "dim",
                )
            )
            if remaining <= 0:
                timer.stop()

        timer = self.set_interval(0.5, tick)
        tick()

    def _update_subtitle(self) -> None:
        base = f"{self.provider_name}/{self.model_name}"
        self.sub_title = (
            f"{base} · {self._context_pct}% context"
            if self._context_pct is not None
            else base
        )

    def _on_tool_call(self, name: str, args: dict) -> None:
        self.call_from_thread(self._log, format_tool_call(name, args))

    def _on_tool_result(self, name: str, result: str, is_error: bool) -> None:
        if len(result) < 500:
            preview = result
        else:
            hidden = len(result) - 500
            preview = f"{result[:500]}… (+{hidden} chars hidden)"
        if result == "User declined to run this tool.":
            self.call_from_thread(self._log, "  [dim]⊘ skipped[/dim]")
        elif result.startswith("User declined and said: "):
            said = result[len("User declined and said: ") :]
            self.call_from_thread(
                self._log,
                self._styled(
                    "  [bold white]↪ redirected:[/bold white] ", said
                ),
            )
        elif is_error:
            self.call_from_thread(
                self._log,
                self._styled("  [bold white]✗[/bold white] ", preview, "dim"),
            )
        else:
            self.call_from_thread(
                self._log,
                self._styled("  [bold white]✓[/bold white] ", preview, "dim"),
            )

    def _on_text_delta(self, delta: str) -> None:
        self._text_buffer += delta

    def _on_usage(self, _turn: Usage, total: Usage, elapsed: float) -> None:
        self.call_from_thread(self._stop_thinking)
        if self._text_buffer.strip():
            self.call_from_thread(
                self._log,
                Panel(
                    Markdown(self._text_buffer),
                    border_style="grey50",
                    subtitle=(
                        "[bold white on grey30] Agent [/bold white on grey30]"
                    ),
                    subtitle_align="right",
                ),
            )
        self._text_buffer = ""
        tokens = (
            format_tokens(total.input_tokens)
            + " in / "
            + format_tokens(total.output_tokens)
            + " out"
        )
        self.call_from_thread(
            self._log, f"[dim]🤖 {tokens} · {elapsed:.1f}s[/dim]"
        )

    def _show_confirm_prompt(self, name: str, preview: str) -> None:
        self._log(
            Panel(
                render_preview(preview),
                title=f"{tool_icon(name)} run {name}?",
                subtitle="[dim]Enter/y = run · n = skip[/dim]",
                border_style="grey50",
            )
        )

    CONFIRM_TIMEOUT = 3.0
    _STALE_ANSWER_GRACE = 10.0

    def _drain_stale_confirm_answer(self) -> None:
        """A confirm that timed out leaves the reader still waiting for
        that line — if it lands after we've already moved on to the
        *next* confirm, it would otherwise be misread as that prompt's
        answer. Discard it here instead, once, within a grace window."""
        if not self._stale_confirm_answer:
            return
        if time.monotonic() > self._stale_expires_at:
            self._stale_confirm_answer = False
            return
        try:
            self._confirm_queue.get_nowait()
        except queue.Empty:
            return
        self._stale_confirm_answer = False

    def _confirm(self, name: str, preview: str) -> bool | str:
        if self._auto_approve:
            self.call_from_thread(self._log, "  [dim]running…[/dim]")
            return True
        self._drain_stale_confirm_answer()
        self._awaiting_confirm = True
        self.call_from_thread(self._show_confirm_prompt, name, preview)
        try:
            answer = self._confirm_queue.get(timeout=self.CONFIRM_TIMEOUT)
        except queue.Empty:
            self._awaiting_confirm = False
            self._stale_confirm_answer = True
            self._stale_expires_at = (
                time.monotonic() + self._STALE_ANSWER_GRACE
            )
            self.call_from_thread(
                self._log,
                "[dim]⏱ no response — running automatically[/dim]",
            )
            return True
        self._awaiting_confirm = False
        text = answer.strip()
        lowered = text.lower()
        if lowered in {"", "y", "yes"}:
            return True
        if lowered in {"n", "no"}:
            return False
        return text
