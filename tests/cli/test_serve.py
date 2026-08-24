"""Unit tests for RpcServer's resilience against a disconnected client
and an unresponsive one — no subprocess, no real stdin/stdout, so these
stay fast and don't share the flakiness of the end-to-end serve tests."""

import io
import json
import queue
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from pycodeloop.cli.serve import RpcServer


def _fake_server():
    agent = SimpleNamespace(
        on_request=None,
        on_text_delta=None,
        on_turn_end=None,
        on_tool_call=None,
        on_tool_result=None,
        on_usage=None,
        on_context=None,
        on_retry=None,
        on_compact_start=None,
        on_compact_end=None,
        confirm=None,
    )
    flow = SimpleNamespace(agent=agent, config=SimpleNamespace(storage=None))
    return RpcServer(flow, "generic", "my-model")


class TestSendBrokenPipe(unittest.TestCase):
    def test_broken_pipe_marks_disconnected_instead_of_raising(self):
        server = _fake_server()

        with mock.patch(
            "sys.stdout", new=mock.Mock(write=mock.Mock(side_effect=BrokenPipeError))
        ):
            server._send({"jsonrpc": "2.0", "method": "chat/heartbeat", "params": {}})

        self.assertTrue(server._disconnected)

    def test_further_sends_are_skipped_once_disconnected(self):
        server = _fake_server()
        server._disconnected = True
        stdout = mock.Mock()

        with mock.patch("sys.stdout", new=stdout):
            server._send({"jsonrpc": "2.0", "method": "chat/heartbeat", "params": {}})

        stdout.write.assert_not_called()


class TestConfirmTimeout(unittest.TestCase):
    def test_confirm_times_out_and_declines(self):
        server = _fake_server()
        server._send = mock.Mock()

        import pycodeloop.cli.serve as serve_module

        with mock.patch.object(serve_module, "_CONFIRM_TIMEOUT", 0.05):
            server._wire_callbacks()
            answer = server.flow.agent.confirm("bash", "$ echo hi")

        self.assertFalse(answer)
        methods = [call.args[0]["method"] for call in server._send.call_args_list]
        self.assertIn("chat/confirmTimeout", methods)

    def test_confirm_returns_answer_when_it_arrives_in_time(self):
        server = _fake_server()
        server._send = mock.Mock(
            side_effect=lambda msg: (
                server._confirm_waiters[msg["params"]["id"]].put(True)
                if msg["method"] == "chat/confirmRequest"
                else None
            )
        )
        server._wire_callbacks()

        answer = server.flow.agent.confirm("bash", "$ echo hi")

        self.assertTrue(answer)


class TestMalformedInputLine(unittest.TestCase):
    def test_malformed_line_is_logged_and_skipped_not_silently_dropped(self):
        server = _fake_server()
        server._send = mock.Mock()
        server.handle = mock.Mock()

        with mock.patch(
            "sys.stdin", new=io.StringIO("not json at all\n")
        ), mock.patch("pycodeloop.cli.serve.console.print") as mock_print:
            server.serve_forever()

        mock_print.assert_called_once()
        server.handle.assert_not_called()
