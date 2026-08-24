"""Test CodeLoopApp's slash-command handling and confirm-queue staleness
(unit-level, without spinning up the full Textual app)."""

import time
import unittest
from types import SimpleNamespace
from unittest import mock

from rich.markdown import Markdown
from rich.panel import Panel

from pycodeloop.cli.chat import CodeLoopApp


def _fake_app(model="claude-sonnet-5", reloadable=False):
    provider = SimpleNamespace(model=model)
    if reloadable:
        provider.reload = mock.Mock(
            side_effect=lambda: setattr(provider, "model", "reloaded-model")
        )
    flow = SimpleNamespace(agent=SimpleNamespace(provider=provider))
    app = CodeLoopApp(flow, "generic", model)
    app._log = mock.Mock()
    app.call_from_thread = lambda fn, *a, **k: fn(*a, **k)
    return app


class TestHandleCommand(unittest.TestCase):
    def test_not_a_command_returns_false(self):
        app = _fake_app()

        self.assertFalse(app._handle_command("hello there"))

    def test_model_alone_reports_current_model(self):
        app = _fake_app(model="claude-sonnet-5")

        self.assertTrue(app._handle_command("/model"))
        self.assertEqual(app.flow.agent.provider.model, "claude-sonnet-5")

    def test_model_with_arg_switches_model_and_subtitle(self):
        app = _fake_app(model="claude-sonnet-5")

        self.assertTrue(app._handle_command("/model gpt-5"))

        self.assertEqual(app.flow.agent.provider.model, "gpt-5")
        self.assertEqual(app.model_name, "gpt-5")
        self.assertIn("gpt-5", app.sub_title)

    def test_reload_calls_provider_reload(self):
        app = _fake_app(model="old-model", reloadable=True)

        self.assertTrue(app._handle_command("/reload"))

        app.flow.agent.provider.reload.assert_called_once()
        self.assertEqual(app.model_name, "reloaded-model")

    def test_reload_without_support_is_a_noop(self):
        app = _fake_app(reloadable=False)

        self.assertTrue(app._handle_command("/reload"))


class TestConfirmStaleness(unittest.TestCase):
    def test_timeout_auto_confirms_and_marks_stale(self):
        app = _fake_app()
        app.CONFIRM_TIMEOUT = 0.05

        result = app._confirm("bash", "$ echo hi")

        self.assertTrue(result)
        self.assertTrue(app._stale_confirm_answer)

    def test_late_answer_does_not_leak_into_next_confirm(self):
        app = _fake_app()
        app.CONFIRM_TIMEOUT = 0.05

        first = app._confirm("bash", "$ echo a")
        self.assertTrue(first)

        # Late answer for the first prompt arrives after its timeout,
        # before a second confirm gets a chance to ask.
        app._confirm_queue.put("n")

        second = app._confirm("bash", "$ echo b")

        self.assertTrue(
            second,
            "stale answer from the first prompt leaked into the second "
            "confirm instead of being drained",
        )

    def test_plain_answer_still_works_normally(self):
        app = _fake_app()
        app._confirm_queue.put("n")

        result = app._confirm("bash", "$ echo hi")

        self.assertFalse(result)

    def test_freeform_text_answer_is_returned_as_redirect(self):
        app = _fake_app()
        app._confirm_queue.put("use ls instead")

        result = app._confirm("bash", "$ rm -rf /tmp/x")

        self.assertEqual(result, "use ls instead")

    def test_expired_stale_flag_is_not_drained_forever(self):
        app = _fake_app()
        app._stale_confirm_answer = True
        app._stale_expires_at = time.monotonic() - 1  # already expired

        app._drain_stale_confirm_answer()

        self.assertFalse(app._stale_confirm_answer)


class TestRunTurnPreservesStreamedText(unittest.TestCase):
    """`_text_buffer` accumulates streamed text as it arrives; if
    `flow.run()` then raises (a failure not covered by the provider's
    own partial-response handling), that already-streamed text used to
    vanish — only the generic error was shown. It must now be flushed
    to the log first."""

    def test_partial_text_is_logged_before_the_error(self):
        app = _fake_app()

        def failing_run(*args, **kwargs):
            app._text_buffer = "here is what I had so far"
            raise RuntimeError("connection died")

        app.flow.run = failing_run

        import asyncio

        asyncio.run(app._run_turn("do something"))

        logged = [call.args[0] for call in app._log.call_args_list]
        markdown_bodies = [
            entry.renderable.markup
            for entry in logged
            if isinstance(entry, Panel)
            and isinstance(entry.renderable, Markdown)
        ]
        self.assertTrue(
            any(
                "here is what I had so far" in body for body in markdown_bodies
            )
        )
        self.assertEqual(app._text_buffer, "")

    def test_no_buffer_only_logs_the_error(self):
        app = _fake_app()
        app.flow.run = mock.Mock(side_effect=RuntimeError("boom"))

        import asyncio

        asyncio.run(app._run_turn("do something"))

        logged = [call.args[0] for call in app._log.call_args_list]
        self.assertTrue(any("boom" in str(entry) for entry in logged))
        self.assertFalse(
            any("interrupted" in str(entry).lower() for entry in logged)
        )


if __name__ == "__main__":
    unittest.main()
