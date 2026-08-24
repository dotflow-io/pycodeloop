"""Test Agent class"""

import threading
import time
import unittest
from unittest import mock

from pycodeloop.abc.confirm import Confirm
from pycodeloop.abc.provider import Provider, ProviderResponse, ToolCall, Usage
from pycodeloop.abc.tool import Tool, ToolResult
from pycodeloop.core.agent import Agent
from pycodeloop.core.session import Session


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, scripted: list[ProviderResponse]) -> None:
        super().__init__(model="fake-model")
        self._scripted = list(scripted)

    def complete(
        self, system_prompt, messages, tools, on_delta=None, cancel_event=None
    ) -> ProviderResponse:
        return self._scripted.pop(0)


class EchoTool(Tool):
    name = "echo"
    description = "Echo the given text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, text: str) -> ToolResult:
        return ToolResult(output=f"echo: {text}")


class FailTool(Tool):
    name = "fail"
    description = "Always fails."

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(output="boom", is_error=True)


class DeleteTool(Tool):
    name = "delete_everything"
    description = "Deletes everything."
    dangerous = True

    def run(self, **kwargs) -> ToolResult:
        return ToolResult(output="deleted")


class StrictPreviewTool(Tool):
    """Mirrors GitCommitTool's real shape: a dangerous tool whose
    preview() requires an argument the model can fail to supply."""

    name = "strict_preview"
    description = "Dangerous tool with a required preview() argument."
    dangerous = True

    def preview(self, message: str, **_) -> str:
        return f"$ do it: {message}"

    def run(self, message: str, **_) -> ToolResult:
        return ToolResult(output=f"did it: {message}")


class TestAgent(unittest.TestCase):
    def test_refuses_dangerous_tool_without_confirm(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="1", name="delete_everything", arguments={}
                        )
                    ],
                ),
                ProviderResponse(text="blocked"),
            ]
        )
        agent = Agent(provider=provider, tools=[DeleteTool()])

        result = agent.run("wipe")

        self.assertEqual(result, "blocked")

    def test_runs_dangerous_tool_when_confirm_approves(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="1", name="delete_everything", arguments={}
                        )
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        agent = Agent(
            provider=provider,
            tools=[DeleteTool()],
            confirm=lambda *_args: True,
        )

        result = agent.run("wipe")

        self.assertEqual(result, "done")

    def test_returns_text_when_no_tool_calls(self):
        provider = FakeProvider([ProviderResponse(text="hello")])
        agent = Agent(provider=provider, tools=[EchoTool()])

        result = agent.run("hi")

        self.assertEqual(result, "hello")

    def test_retries_and_recovers_from_empty_response_with_no_tool_calls(
        self,
    ):
        """Regression: a weak model can return a fully empty message (no
        text, no tool calls) — the agent used to treat that as a normal
        `done` turn and silently return "". It should retry instead."""
        provider = FakeProvider(
            [
                ProviderResponse(text="", tool_calls=[]),
                ProviderResponse(text="", tool_calls=[]),
                ProviderResponse(text="finally, here's the answer"),
            ]
        )
        agent = Agent(provider=provider, tools=[EchoTool()])

        result = agent.run("hi")

        self.assertEqual(result, "finally, here's the answer")

    def test_gives_up_with_a_clear_error_after_repeated_empty_responses(
        self,
    ):
        provider = FakeProvider(
            [ProviderResponse(text="", tool_calls=[]) for _ in range(5)]
        )
        agent = Agent(provider=provider, tools=[EchoTool()])

        result = agent.run("hi")

        self.assertIn("empty response", result)
        self.assertIn(provider.model, result)

    def test_accumulates_usage_across_empty_response_retries(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="", tool_calls=[], usage=Usage(input_tokens=10)
                ),
                ProviderResponse(
                    text="", tool_calls=[], usage=Usage(input_tokens=20)
                ),
                ProviderResponse(text="done", usage=Usage(input_tokens=30)),
            ]
        )
        agent = Agent(provider=provider, tools=[EchoTool()])

        agent.run("hi")

        self.assertEqual(agent.usage.input_tokens, 60)

    def test_accumulates_usage_even_when_giving_up_after_empty_responses(
        self,
    ):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="", tool_calls=[], usage=Usage(input_tokens=5)
                )
                for _ in range(3)
            ]
        )
        agent = Agent(provider=provider, tools=[EchoTool()])

        agent.run("hi")

        self.assertEqual(agent.usage.input_tokens, 15)

    def test_records_assistant_turn_in_session_when_giving_up(self):
        provider = FakeProvider(
            [ProviderResponse(text="", tool_calls=[]) for _ in range(3)]
        )
        notified = []
        agent = Agent(
            provider=provider,
            tools=[EchoTool()],
            on_message=lambda: notified.append(True),
        )
        session = Session(system_prompt="sys")

        result = agent.run("hi", session=session)

        history = session.history()
        self.assertEqual(history[-1].role, "assistant")
        self.assertEqual(history[-1].content, result)
        self.assertTrue(notified)

    def test_prefers_explicit_provider_context_window(self):
        provider = FakeProvider(
            [ProviderResponse(text="done", usage=Usage(input_tokens=90))]
        )
        provider.context_window = 100
        contexts = []
        agent = Agent(
            provider=provider,
            on_context=lambda used, limit: contexts.append((used, limit)),
        )

        result = agent.run("hi")

        self.assertEqual(result, "done")
        self.assertEqual(contexts, [(90, 100)])

    def test_executes_tool_call_then_returns_final_text(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="1", name="echo", arguments={"text": "hey"}
                        )
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        calls = []
        agent = Agent(
            provider=provider,
            tools=[EchoTool()],
            on_tool_result=lambda name, result, is_error: calls.append(
                (name, result, is_error)
            ),
        )

        result = agent.run("do it")

        self.assertEqual(result, "done")
        self.assertEqual(calls, [("echo", "echo: hey", False)])

    def test_reports_unknown_tool(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="missing", arguments={})
                    ],
                ),
                ProviderResponse(text="ok"),
            ]
        )
        results = []
        agent = Agent(
            provider=provider,
            tools=[EchoTool()],
            on_tool_result=lambda _name, result, _is_error: results.append(
                result
            ),
        )

        agent.run("do it")

        self.assertEqual(results, ["Unknown tool: missing"])

    def test_reports_tool_is_error_flag(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[ToolCall(id="1", name="fail", arguments={})],
                ),
                ProviderResponse(text="ok"),
            ]
        )
        calls = []
        agent = Agent(
            provider=provider,
            tools=[FailTool()],
            on_tool_result=lambda name, result, is_error: calls.append(
                (name, result, is_error)
            ),
        )

        agent.run("do it")

        self.assertEqual(calls, [("fail", "boom", True)])

    def test_preview_raising_reports_is_error_instead_of_crashing_the_turn(
        self,
    ):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="strict_preview", arguments={})
                    ],
                ),
                ProviderResponse(text="ok"),
            ]
        )
        calls = []
        agent = Agent(
            provider=provider,
            tools=[StrictPreviewTool()],
            confirm=lambda *_args: True,
            on_tool_result=lambda name, result, is_error: calls.append(
                (name, result, is_error)
            ),
        )

        result = agent.run("do it")

        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)
        name, message, is_error = calls[0]
        self.assertEqual(name, "strict_preview")
        self.assertTrue(is_error)
        self.assertIn("missing 1 required positional argument", message)

    def test_skips_dangerous_tool_when_confirm_declines(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="1", name="delete_everything", arguments={}
                        )
                    ],
                ),
                ProviderResponse(text="ok"),
            ]
        )
        results = []
        agent = Agent(
            provider=provider,
            tools=[DeleteTool()],
            confirm=lambda _name, _preview: False,
            on_tool_result=lambda _name, result, _is_error: results.append(
                result
            ),
        )

        agent.run("do it")

        self.assertEqual(results, ["User declined to run this tool."])

    def test_runs_dangerous_tool_when_confirm_accepts(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="1", name="delete_everything", arguments={}
                        )
                    ],
                ),
                ProviderResponse(text="ok"),
            ]
        )
        results = []
        agent = Agent(
            provider=provider,
            tools=[DeleteTool()],
            confirm=lambda _name, _preview: True,
            on_tool_result=lambda _name, result, _is_error: results.append(
                result
            ),
        )

        agent.run("do it")

        self.assertEqual(results, ["deleted"])

    def test_accepts_confirm_abc_instance_instead_of_callable(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="1", name="delete_everything", arguments={}
                        )
                    ],
                ),
                ProviderResponse(text="ok"),
            ]
        )

        class AlwaysConfirm(Confirm):
            def ask(self, name: str, preview: str) -> bool | str:
                return True

        results = []
        agent = Agent(
            provider=provider,
            tools=[DeleteTool()],
            confirm=AlwaysConfirm(),
            on_tool_result=lambda _name, result, _is_error: results.append(
                result
            ),
        )

        agent.run("do it")

        self.assertEqual(results, ["deleted"])

    def test_on_request_fires_before_each_provider_call(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="echo", arguments={"text": "x"})
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        calls = []
        agent = Agent(
            provider=provider,
            tools=[EchoTool()],
            on_request=lambda messages, tools: calls.append((messages, tools)),
        )

        agent.run("do it")

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], (1, 1))

    def test_max_history_turns_trims_session_before_each_run(self):
        session = Session(system_prompt="sys")
        session.add_user("old-1")
        session.add_assistant("old-1-reply")
        session.add_user("old-2")
        session.add_assistant("old-2-reply")

        provider = FakeProvider([ProviderResponse(text="new-reply")])
        agent = Agent(provider=provider, max_history_turns=1)

        agent.run("new", session=session)

        contents = [m.content for m in session.messages]
        self.assertEqual(contents, ["new", "new-reply"])

    def test_no_max_history_turns_leaves_session_unbounded(self):
        session = Session(system_prompt="sys")
        session.add_user("old-1")
        session.add_assistant("old-1-reply")

        provider = FakeProvider([ProviderResponse(text="new-reply")])
        agent = Agent(provider=provider)

        agent.run("new", session=session)

        self.assertEqual(len(session.messages), 4)

    def test_on_usage_reports_elapsed_time(self):
        provider = FakeProvider([ProviderResponse(text="hello")])
        seen = []
        agent = Agent(
            provider=provider,
            on_usage=lambda _turn, _total, elapsed: seen.append(elapsed),
        )

        agent.run("hi")

        self.assertEqual(len(seen), 1)
        self.assertGreaterEqual(seen[0], 0)

    def test_on_context_reports_tokens_against_known_window(self):
        provider = FakeProvider(
            [ProviderResponse(text="hi", usage=Usage(input_tokens=42))]
        )
        provider.model = "claude-sonnet-5"
        seen = []
        agent = Agent(
            provider=provider, on_context=lambda *args: seen.append(args)
        )

        agent.run("hi")

        self.assertEqual(seen, [(42, 200_000)])

    def test_compacts_when_context_usage_crosses_threshold(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="reply-1", usage=Usage(input_tokens=190_000)
                ),
                ProviderResponse(
                    text="reply-2", usage=Usage(input_tokens=190_000)
                ),
                ProviderResponse(text="summary of earlier turns"),
                ProviderResponse(
                    text="reply-3", usage=Usage(input_tokens=1_000)
                ),
            ]
        )
        provider.model = "claude-sonnet-5"
        events = []
        agent = Agent(
            provider=provider,
            on_compact_start=lambda: events.append("start"),
            on_compact_end=lambda before, after: events.append(
                (before, after)
            ),
        )
        session = Session(system_prompt="sys")

        agent.run("first", session=session)
        agent.run("second", session=session)
        agent.run("third", session=session)

        self.assertEqual(events, ["start", (5, 3)])
        self.assertEqual(session.messages[0].role, "user")
        self.assertIn("summary of earlier turns", session.messages[0].content)
        self.assertIn("second", session.messages[0].content)
        self.assertEqual(len(session.messages), 4)

    def test_context_usage_does_not_leak_across_sessions(self):
        provider = FakeProvider(
            [
                ProviderResponse(text="b1", usage=Usage(input_tokens=500)),
                ProviderResponse(text="b2", usage=Usage(input_tokens=500)),
                ProviderResponse(text="a1", usage=Usage(input_tokens=190_000)),
                ProviderResponse(text="b3", usage=Usage(input_tokens=500)),
            ]
        )
        provider.model = "claude-sonnet-5"
        events = []
        agent = Agent(
            provider=provider,
            on_compact_start=lambda: events.append("start"),
        )
        session_a = Session(system_prompt="sys")
        session_b = Session(system_prompt="sys")

        agent.run("b-first", session=session_b)
        agent.run("b-second", session=session_b)
        agent.run("a-first", session=session_a)
        agent.run("b-third", session=session_b)

        self.assertEqual(events, [])
        self.assertEqual(len(session_b.messages), 6)

    def test_auto_compact_false_never_compacts(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="reply-1", usage=Usage(input_tokens=190_000)
                ),
                ProviderResponse(
                    text="reply-2", usage=Usage(input_tokens=190_000)
                ),
                ProviderResponse(
                    text="reply-3", usage=Usage(input_tokens=190_000)
                ),
            ]
        )
        provider.model = "claude-sonnet-5"
        agent = Agent(provider=provider, auto_compact=False)
        session = Session(system_prompt="sys")

        agent.run("first", session=session)
        agent.run("second", session=session)
        agent.run("third", session=session)

        self.assertEqual(len(session.messages), 6)

    def test_cancel_event_stops_before_the_next_provider_call(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="echo", arguments={"text": "a"})
                    ],
                ),
                ProviderResponse(text="should never be reached"),
            ]
        )
        cancel_event = threading.Event()

        class CancellingEchoTool(EchoTool):
            def run(self, text: str) -> ToolResult:
                cancel_event.set()
                return super().run(text)

        agent = Agent(provider=provider, tools=[CancellingEchoTool()])
        session = Session(system_prompt="sys")

        result = agent.run("go", session=session, cancel_event=cancel_event)

        self.assertEqual(result, "Cancelled by user.")
        self.assertEqual(
            len(provider._scripted), 1
        )  # 2nd response never consumed
        roles = [m.role for m in session.messages]
        self.assertEqual(roles, ["user", "assistant", "tool"])

    def test_cancel_event_set_before_run_returns_immediately(self):
        provider = FakeProvider(
            [ProviderResponse(text="should never be reached")]
        )
        cancel_event = threading.Event()
        cancel_event.set()
        agent = Agent(provider=provider)

        result = agent.run("go", cancel_event=cancel_event)

        self.assertEqual(result, "Cancelled by user.")
        self.assertEqual(len(provider._scripted), 1)

    def test_cancelled_provider_response_stops_the_run_immediately(self):
        """Regression: cancel_event was never threaded into
        provider.complete(), so cancelling mid-stream had no effect
        until the current turn finished on its own. Once the provider
        reports stop_reason='cancelled', the run must stop right away
        instead of retrying or continuing to the next turn."""

        class CancelAwareProvider(Provider):
            name = "cancel-aware"

            def __init__(self) -> None:
                super().__init__(model="fake-model")

            def complete(
                self,
                system_prompt,
                messages,
                tools,
                on_delta=None,
                cancel_event=None,
            ) -> ProviderResponse:
                assert cancel_event is not None, (
                    "cancel_event was not forwarded to provider.complete()"
                )
                cancel_event.set()
                return ProviderResponse(
                    text="partial answer", stop_reason="cancelled"
                )

        provider = CancelAwareProvider()
        cancel_event = threading.Event()
        agent = Agent(provider=provider, tools=[EchoTool()])
        session = Session(system_prompt="sys")

        result = agent.run("go", session=session, cancel_event=cancel_event)

        self.assertEqual(result, "Cancelled by user.")
        self.assertEqual(session.messages[-1].content, "partial answer")


class FlakyProvider(Provider):
    """Raises a retryable error `fail_times` times, then succeeds."""

    name = "flaky"

    def __init__(self, fail_times: int, status_code: int = 429) -> None:
        super().__init__(model="fake-model")
        self.fail_times = fail_times
        self.status_code = status_code
        self.calls = 0

    def complete(
        self, system_prompt, messages, tools, on_delta=None, cancel_event=None
    ) -> ProviderResponse:
        self.calls += 1
        if self.calls <= self.fail_times:
            exc = RuntimeError(f"failed attempt {self.calls}")
            exc.status_code = self.status_code
            raise exc
        return ProviderResponse(text="ok")


class TestAgentRetry(unittest.TestCase):
    def setUp(self):
        import pycodeloop.core.agent as agent_module

        patcher = mock.patch.object(agent_module.time, "sleep")
        self.mock_sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def test_retries_retryable_errors_and_eventually_succeeds(self):
        provider = FlakyProvider(fail_times=2)
        events = []
        agent = Agent(
            provider=provider,
            tools=[],
            on_retry=lambda attempt, delay, _exc: events.append(
                (attempt, delay)
            ),
        )

        result = agent.run("hi")

        self.assertEqual(result, "ok")
        self.assertEqual(provider.calls, 3)
        self.assertEqual(events, [(1, 1.0), (2, 2.0)])

    def test_gives_up_after_max_retries(self):
        provider = FlakyProvider(fail_times=10)
        agent = Agent(provider=provider, tools=[])

        with self.assertRaises(RuntimeError):
            agent.run("hi")

        self.assertEqual(provider.calls, 4)  # 1 initial + 3 retries

    def test_non_retryable_error_raises_immediately(self):
        provider = FlakyProvider(fail_times=1, status_code=400)
        agent = Agent(provider=provider, tools=[])

        with self.assertRaises(RuntimeError):
            agent.run("hi")

        self.assertEqual(provider.calls, 1)

    def test_falls_back_to_next_provider_after_retries_exhausted(self):
        primary = FlakyProvider(fail_times=10)
        secondary = FlakyProvider(fail_times=0)
        agent = Agent(
            provider=primary,
            fallback_providers=[secondary],
            tools=[],
        )

        result = agent.run("hi")

        self.assertEqual(result, "ok")
        self.assertEqual(primary.calls, 4)  # 1 initial + 3 retries
        self.assertEqual(secondary.calls, 1)
        self.assertIs(agent.provider, secondary)

    def test_falls_back_immediately_on_non_retryable_error(self):
        primary = FlakyProvider(fail_times=1, status_code=400)
        secondary = FlakyProvider(fail_times=0)
        agent = Agent(
            provider=primary,
            fallback_providers=[secondary],
            tools=[],
        )

        result = agent.run("hi")

        self.assertEqual(result, "ok")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(secondary.calls, 1)

    def test_raises_when_every_provider_in_the_chain_fails(self):
        primary = FlakyProvider(fail_times=10)
        secondary = FlakyProvider(fail_times=10)
        agent = Agent(
            provider=primary,
            fallback_providers=[secondary],
            tools=[],
        )

        with self.assertRaises(RuntimeError):
            agent.run("hi")

        self.assertEqual(primary.calls, 4)
        self.assertEqual(secondary.calls, 4)

    def test_a_new_run_call_retries_the_primary_even_after_a_prior_fallback(
        self,
    ):
        """Issue #18: falling back within one run() must not be permanent
        across the agent's lifetime — a later run() call (a new user
        message) should give the primary another chance instead of
        staying stuck on the fallback forever."""
        primary = FlakyProvider(fail_times=4)
        secondary = FlakyProvider(fail_times=0)
        agent = Agent(
            provider=primary,
            fallback_providers=[secondary],
            tools=[],
        )

        first = agent.run("hi")
        self.assertEqual(first, "ok")
        self.assertIs(agent.provider, secondary)
        self.assertEqual(primary.calls, 4)
        self.assertEqual(secondary.calls, 1)

        second = agent.run("hi again")

        self.assertEqual(second, "ok")
        self.assertIs(agent.provider, primary)
        self.assertEqual(primary.calls, 5)
        self.assertEqual(secondary.calls, 1)

    def test_second_turn_after_fallback_does_not_retry_the_dead_primary(self):
        """Regression: rebuilding `[self.provider, *self.fallback_providers]`
        fresh on every `_complete` call put the now-active fallback in the
        list twice (once as `self.provider`, once at its original index)
        and never retried a provider earlier in the chain — the fixed
        `_provider_chain` walked by index must instead pick up exactly
        where the previous call left off."""
        primary = FlakyProvider(fail_times=10)
        secondary = FlakyProvider(fail_times=0)
        agent = Agent(
            provider=primary,
            fallback_providers=[secondary],
            tools=[],
        )

        agent._complete(system_prompt="s", messages=[], tools=[])
        self.assertEqual(primary.calls, 4)
        self.assertEqual(secondary.calls, 1)

        agent._complete(system_prompt="s", messages=[], tools=[])
        self.assertEqual(primary.calls, 4)
        self.assertEqual(secondary.calls, 2)

    def test_on_provider_fallback_fires_with_old_new_and_error(self):
        primary = FlakyProvider(fail_times=10)
        secondary = FlakyProvider(fail_times=0)
        events = []
        agent = Agent(
            provider=primary,
            fallback_providers=[secondary],
            tools=[],
            on_provider_fallback=lambda old, new, exc: events.append(
                (old, new, str(exc))
            ),
        )

        agent.run("hi")

        self.assertEqual(len(events), 1)
        old, new, error = events[0]
        self.assertIs(old, primary)
        self.assertIs(new, secondary)
        self.assertIn("failed attempt", error)


class TestAgentParallelTools(unittest.TestCase):
    def test_independent_safe_tools_run_concurrently(self):
        barrier = threading.Barrier(2, timeout=2)

        class BarrierTool(Tool):
            name = "barrier"
            description = "waits at a barrier to prove concurrency"
            parameters = {
                "type": "object",
                "properties": {"id": {"type": "string"}},
            }

            def run(self, id: str) -> ToolResult:
                barrier.wait()
                return ToolResult(output=f"done {id}")

        class BarrierTool2(BarrierTool):
            name = "barrier2"

        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="1", name="barrier", arguments={"id": "a"}
                        ),
                        ToolCall(
                            id="2", name="barrier2", arguments={"id": "b"}
                        ),
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        agent = Agent(provider=provider, tools=[BarrierTool(), BarrierTool2()])

        result = agent.run("go")

        self.assertEqual(result, "done")

    def test_dangerous_tool_in_batch_forces_sequential_and_confirms(self):
        class SafeTool(Tool):
            name = "safe"
            description = "safe"
            parameters = {"type": "object", "properties": {}}

            def run(self) -> ToolResult:
                return ToolResult(output="safe done")

        class DangerTool(Tool):
            name = "danger"
            description = "danger"
            parameters = {"type": "object", "properties": {}}
            dangerous = True

            def run(self) -> ToolResult:
                return ToolResult(output="danger done")

        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="safe", arguments={}),
                        ToolCall(id="2", name="danger", arguments={}),
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        confirmed = []
        agent = Agent(
            provider=provider,
            tools=[SafeTool(), DangerTool()],
            confirm=lambda name, _preview: confirmed.append(name) or True,
        )

        agent.run("go")

        self.assertEqual(confirmed, ["danger"])

    def test_repeated_tool_name_in_batch_runs_sequential(self):
        calls_seen = []

        class RecordingTool(Tool):
            name = "rec"
            description = "records call order, not thread-safe"
            parameters = {
                "type": "object",
                "properties": {"n": {"type": "string"}},
            }

            def run(self, n: str) -> ToolResult:
                calls_seen.append(n)
                return ToolResult(output=f"ok {n}")

        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="rec", arguments={"n": "1"}),
                        ToolCall(id="2", name="rec", arguments={"n": "2"}),
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        agent = Agent(provider=provider, tools=[RecordingTool()])

        agent.run("go")

        self.assertEqual(calls_seen, ["1", "2"])

    def test_repeated_concurrent_safe_tool_name_runs_in_parallel(self):
        """A tool that opts into concurrent_safe (e.g. DelegateTool) must
        run its repeated same-name calls in parallel, or fanning out N
        sub-agents in one turn would serialize them — defeating the point."""
        barrier = threading.Barrier(2, timeout=2)

        class ConcurrentSafeTool(Tool):
            name = "spawn"
            description = "opts into concurrent same-name execution"
            parameters = {
                "type": "object",
                "properties": {"id": {"type": "string"}},
            }
            concurrent_safe = True

            def run(self, id: str) -> ToolResult:
                barrier.wait()
                return ToolResult(output=f"done {id}")

        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="spawn", arguments={"id": "a"}),
                        ToolCall(id="2", name="spawn", arguments={"id": "b"}),
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        agent = Agent(provider=provider, tools=[ConcurrentSafeTool()])

        result = agent.run("go")

        self.assertEqual(result, "done")

    def test_tool_opting_into_cancel_event_receives_the_agents_event(self):
        received = []

        class CancelAwareTool(Tool):
            name = "cancel_aware"
            description = "wants the agent's cancel_event"
            parameters = {"type": "object", "properties": {}}
            wants_cancel_event = True

            def run(self, cancel_event=None) -> ToolResult:
                received.append(cancel_event)
                return ToolResult(output="ok")

        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="cancel_aware", arguments={})
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        agent = Agent(provider=provider, tools=[CancelAwareTool()])
        event = threading.Event()

        agent.run("go", cancel_event=event)

        self.assertEqual(received, [event])

    def test_slow_tool_in_a_parallel_batch_times_out_without_blocking_the_others(
        self,
    ):
        class SlowTool(Tool):
            name = "slow"
            description = "never finishes within its timeout"
            parameters = {"type": "object", "properties": {}}
            timeout = 0.05

            def run(self) -> ToolResult:
                time.sleep(2)
                return ToolResult(output="too late")

        class FastTool(Tool):
            name = "fast"
            description = "finishes immediately"
            parameters = {"type": "object", "properties": {}}

            def run(self) -> ToolResult:
                return ToolResult(output="fast done")

        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="slow", arguments={}),
                        ToolCall(id="2", name="fast", arguments={}),
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        agent = Agent(provider=provider, tools=[SlowTool(), FastTool()])

        started = time.monotonic()
        result = agent.run("go")
        elapsed = time.monotonic() - started

        self.assertEqual(result, "done")
        self.assertLess(elapsed, 1.0)


class TestAgentToolResultSummarization(unittest.TestCase):
    def test_large_tool_result_gets_summarized(self):
        class HugeOutputTool(Tool):
            name = "huge"
            description = "returns huge output"
            parameters = {"type": "object", "properties": {}}

            def run(self) -> ToolResult:
                return ToolResult(output="x" * 10_000)

        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[ToolCall(id="1", name="huge", arguments={})],
                ),
                ProviderResponse(text="condensed"),
                ProviderResponse(text="done"),
            ]
        )
        agent = Agent(provider=provider, tools=[HugeOutputTool()])
        session = Session(system_prompt="sys")

        agent.run("go", session=session)

        tool_message = next(m for m in session.messages if m.role == "tool")
        self.assertLess(len(tool_message.content), 100)
        self.assertIn("condensed", tool_message.content)

    def test_small_tool_result_is_not_summarized(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="echo", arguments={"text": "hi"})
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        agent = Agent(provider=provider, tools=[EchoTool()])
        session = Session(system_prompt="sys")

        agent.run("go", session=session)

        tool_message = next(m for m in session.messages if m.role == "tool")
        self.assertEqual(tool_message.content, "echo: hi")

    def test_error_result_is_not_summarized_even_if_large(self):
        class HugeErrorTool(Tool):
            name = "huge_error"
            description = "returns a huge error"
            parameters = {"type": "object", "properties": {}}

            def run(self) -> ToolResult:
                return ToolResult(output="e" * 10_000, is_error=True)

        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="huge_error", arguments={})
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        agent = Agent(provider=provider, tools=[HugeErrorTool()])
        session = Session(system_prompt="sys")

        agent.run("go", session=session)

        tool_message = next(m for m in session.messages if m.role == "tool")
        self.assertEqual(len(tool_message.content), 10_000)


class TestSafeCallback(unittest.TestCase):
    """A buggy consumer callback (UI rendering, storage persistence,
    etc.) must not abort the turn/tool loop still in progress — only
    that one side effect should fail."""

    def test_on_message_raising_does_not_abort_the_turn(self):
        provider = FakeProvider([ProviderResponse(text="done")])
        agent = Agent(provider=provider, tools=[])
        agent.on_message = mock.Mock(side_effect=RuntimeError("storage down"))
        session = Session(system_prompt="sys")

        result = agent.run("go", session=session)

        self.assertEqual(result, "done")
        self.assertTrue(agent.on_message.called)

    def test_on_tool_result_raising_still_records_the_tool_result(self):
        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="1", name="echo", arguments={"text": "hi"})
                    ],
                ),
                ProviderResponse(text="done"),
            ]
        )
        agent = Agent(provider=provider, tools=[EchoTool()])
        agent.on_tool_result = mock.Mock(
            side_effect=RuntimeError("render bug")
        )
        session = Session(system_prompt="sys")

        result = agent.run("go", session=session)

        self.assertEqual(result, "done")
        tool_message = next(m for m in session.messages if m.role == "tool")
        self.assertEqual(tool_message.content, "echo: hi")

    def test_failing_callback_is_traced_without_raising(self):
        provider = FakeProvider([ProviderResponse(text="done")])
        agent = Agent(provider=provider, tools=[])
        agent.on_usage = mock.Mock(side_effect=ValueError("boom"))
        events = []
        agent.on_trace_event = events.append
        session = Session(system_prompt="sys")

        agent.run("go", session=session)

        callback_errors = [e for e in events if e["type"] == "callback_error"]
        self.assertEqual(len(callback_errors), 1)
        self.assertEqual(callback_errors[0]["error"], "boom")

    def test_trace_event_callback_raising_does_not_propagate(self):
        provider = FakeProvider([ProviderResponse(text="done")])
        agent = Agent(provider=provider, tools=[])
        agent.on_trace_event = mock.Mock(side_effect=RuntimeError("bad sink"))
        session = Session(system_prompt="sys")

        result = agent.run("go", session=session)

        self.assertEqual(result, "done")


if __name__ == "__main__":
    unittest.main()
