"""Test GenericProvider"""

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from pycodeloop.providers import get_provider
from pycodeloop.providers.generic import GenericProvider


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _TimeoutAfterLinesResponse(io.BytesIO):
    """Like `_FakeResponse`, but raises `TimeoutError` once iteration
    passes `raise_after` lines — simulates the model going quiet
    mid-stream (long reasoning) and the socket's read timeout firing
    before any terminal marker or `finish_reason` arrives."""

    def __init__(self, data: bytes, raise_after: int):
        super().__init__(data)
        self._raise_after = raise_after
        self._yielded = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded >= self._raise_after:
            raise TimeoutError("timed out")
        line = super().readline()
        if not line:
            raise StopIteration
        self._yielded += 1
        return line


class GenericProviderTestCase(unittest.TestCase):
    def setUp(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.tmp_path = Path(tmpdir.name)

    def _write_config(self, config: dict) -> Path:
        path = self.tmp_path / "provider.json"
        path.write_text(json.dumps(config))
        return path


class TestLoadProviderFromJson(GenericProviderTestCase):
    def test_inference_params_are_merged_into_the_request_body(self):
        provider = GenericProvider(
            url="http://fake/v1/chat/completions",
            model="my-model",
            inference_params={
                "temperature": 0,
                "max_tokens": 500,
            },
        )

        captured_requests = []

        def fake_urlopen(request, timeout=None):
            captured_requests.append(json.loads(request.data))
            payload = json.dumps(
                {
                    "choices": [
                        {"message": {"content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {},
                }
            ).encode()
            return _FakeResponse(payload)

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.complete("sys", [], [])

        self.assertEqual(captured_requests[0]["temperature"], 0)
        self.assertEqual(captured_requests[0]["max_tokens"], 500)

    def test_inference_params_apply_on_top_of_a_custom_request_builder(self):
        """Config-declared `request.params` are baked into the body once by
        the custom builder — `inference_params` (CLI overrides forwarded via
        `get_provider`) must still reach the wire without doubling them up."""
        path = self._write_config(
            {
                "url": "http://fake/v1/chat/completions",
                "model": "my-model",
                "request": {"params": {"temperature": 0.7}},
            }
        )
        provider = get_provider(
            str(path), inference_params={"temperature": 0.1}
        )

        captured_requests = []

        def fake_urlopen(request, timeout=None):
            captured_requests.append(json.loads(request.data))
            payload = json.dumps(
                {
                    "choices": [
                        {"message": {"content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {},
                }
            ).encode()
            return _FakeResponse(payload)

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.complete("sys", [], [])

        self.assertEqual(captured_requests[0]["temperature"], 0.1)

    def test_builds_generic_provider_from_config(self):
        path = self._write_config(
            {
                "url": "http://fake/v1/chat/completions",
                "model": "my-model",
                "api_key": "sk-test",
                "headers": {"X-Custom": "1"},
            }
        )

        provider = GenericProvider.from_json(path)

        self.assertIsInstance(provider, GenericProvider)
        self.assertEqual(provider.url, "http://fake/v1/chat/completions")
        self.assertEqual(provider.model, "my-model")
        self.assertEqual(provider.api_key, "sk-test")
        self.assertEqual(provider.headers, {"X-Custom": "1"})

    def test_reads_context_window_from_config(self):
        path = self._write_config(
            {
                "url": "http://fake/v1/chat/completions",
                "model": "my-model",
                "context_window": 4096,
            }
        )

        provider = GenericProvider.from_json(path)

        self.assertEqual(provider.context_window, 4096)

    def test_reload_updates_context_window_from_config(self):
        path = self._write_config(
            {
                "url": "http://fake/v1/chat/completions",
                "model": "my-model",
                "context_window": 4096,
            }
        )
        provider = GenericProvider.from_json(path)
        path.write_text(
            json.dumps(
                {
                    "url": "http://fake/v1/chat/completions",
                    "model": "my-model",
                    "context_window": 8192,
                }
            )
        )

        provider.reload()

        self.assertEqual(provider.context_window, 8192)

    def test_reads_api_key_from_env_var(self):
        path = self._write_config(
            {
                "url": "http://fake/v1/chat/completions",
                "model": "my-model",
                "api_key_env": "MY_FAKE_KEY",
            }
        )

        with mock.patch.dict(
            "os.environ", {"MY_FAKE_KEY": "from-env", "PYCODELOOP_API_KEY": ""}
        ):
            provider = GenericProvider.from_json(path)

        self.assertEqual(provider.api_key, "from-env")

    def test_reads_api_key_from_pycodeloop_env_before_named_var(self):
        path = self._write_config(
            {
                "url": "http://fake/v1/chat/completions",
                "model": "my-model",
                "api_key_env": "MY_FAKE_KEY",
            }
        )

        with mock.patch.dict(
            "os.environ",
            {
                "PYCODELOOP_API_KEY": "from-extension",
                "MY_FAKE_KEY": "from-named",
            },
        ):
            provider = GenericProvider.from_json(path)

        self.assertEqual(provider.api_key, "from-extension")

    def test_default_response_shape_without_response_paths(self):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        response_body = json.dumps(
            {
                "choices": [
                    {"message": {"content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        ).encode()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(response_body),
        ):
            result = provider.complete("sys", [], [])

        self.assertEqual(result.text, "hi")

    def test_non_streaming_skips_the_request_when_already_cancelled(self):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)
        cancel_event = threading.Event()
        cancel_event.set()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen"
        ) as urlopen:
            result = provider.complete(
                "sys", [], [], cancel_event=cancel_event
            )

        urlopen.assert_not_called()
        self.assertEqual(result.stop_reason, "cancelled")

    def test_default_response_captures_extra_tool_call_fields(self):
        """Some OpenAI-compatible vendors (e.g. Gemini) attach extra
        sibling fields to a tool_call — like extra_content.google's
        thought_signature — that must round-trip back verbatim on the
        next turn or the vendor rejects the follow-up request."""
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        response_body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "a.py"}',
                                    },
                                    "extra_content": {
                                        "google": {
                                            "thought_signature": "abc123"
                                        }
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {},
            }
        ).encode()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(response_body),
        ):
            result = provider.complete("sys", [], [])

        self.assertEqual(
            result.tool_calls[0].extra,
            {"extra_content": {"google": {"thought_signature": "abc123"}}},
        )

    def test_streaming_response_also_captures_extra_tool_call_fields(self):
        """Same as above, but through _stream() — the code path actually
        used whenever a caller passes on_delta (i.e. every real pycodeloop
        run), which builds ToolCalls from accumulated deltas instead of
        _default_response and previously dropped extra fields entirely."""
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": "",
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"command": "ls"}'
                                    },
                                    "extra_content": {
                                        "google": {
                                            "thought_signature": "xyz789"
                                        }
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        sse_body = (
            "".join(f"data: {json.dumps(c)}\n" for c in chunks)
            + "data: [DONE]\n"
        ).encode()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(sse_body),
        ):
            result = provider.complete("sys", [], [], on_delta=lambda _: None)

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].arguments, {"command": "ls"})
        self.assertEqual(
            result.tool_calls[0].extra,
            {"extra_content": {"google": {"thought_signature": "xyz789"}}},
        )

    def test_streaming_keeps_already_shown_text_on_malformed_chunk(self):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        chunks = [{"choices": [{"delta": {"content": "hello there"}}]}]
        sse_body = (
            "".join(f"data: {json.dumps(c)}\n" for c in chunks)
            + "data: {not valid json\n"
            + "data: [DONE]\n"
        ).encode()

        deltas = []
        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(sse_body),
        ):
            result = provider.complete("sys", [], [], on_delta=deltas.append)

        self.assertEqual(result.text, "hello there")
        self.assertEqual(result.stop_reason, "malformed_stream")
        self.assertEqual("".join(deltas), "hello there")

    def test_streaming_prefers_finish_reason_over_trailing_malformed_chunk(
        self,
    ):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        chunks = [
            {
                "choices": [
                    {"delta": {"content": "done"}, "finish_reason": "stop"}
                ]
            }
        ]
        sse_body = (
            "".join(f"data: {json.dumps(c)}\n" for c in chunks)
            + "data: {junk after finish\n"
        ).encode()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(sse_body),
        ):
            result = provider.complete("sys", [], [], on_delta=lambda _: None)

        self.assertEqual(result.text, "done")
        self.assertEqual(result.stop_reason, "stop")

    def test_streaming_flags_a_connection_dropped_mid_response(self):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        chunks = [{"choices": [{"delta": {"content": "cut off mid"}}]}]
        sse_body = "".join(f"data: {json.dumps(c)}\n" for c in chunks).encode()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(sse_body),
        ):
            result = provider.complete("sys", [], [], on_delta=lambda _: None)

        self.assertEqual(result.text, "cut off mid")
        self.assertEqual(result.stop_reason, "connection_lost")

    def test_streaming_returns_partial_text_on_mid_stream_timeout(self):
        """Regression: a `TimeoutError` raised mid-read (model silent for
        longer than the socket timeout while "thinking") used to
        propagate out of `_stream()` uncaught, discarding whatever text
        had already streamed in and losing the assistant's turn
        entirely instead of returning it as a partial response."""
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        chunks = [
            {"choices": [{"delta": {"content": "partial "}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
        ]
        sse_body = "".join(f"data: {json.dumps(c)}\n" for c in chunks).encode()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_TimeoutAfterLinesResponse(sse_body, raise_after=1),
        ):
            result = provider.complete("sys", [], [], on_delta=lambda _: None)

        self.assertEqual(result.text, "partial ")
        self.assertEqual(result.stop_reason, "connection_lost")

    def test_streaming_reraises_timeout_when_nothing_was_streamed_yet(self):
        """A timeout before any content or tool-call delta arrived means
        nothing was generated to preserve — the exception should still
        propagate so `Agent._complete()`'s existing retry logic kicks
        in, instead of being swallowed into an empty response."""
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        with (
            mock.patch(
                "pycodeloop.providers.generic.urllib.request.urlopen",
                return_value=_TimeoutAfterLinesResponse(b"", raise_after=0),
            ),
            self.assertRaises(TimeoutError),
        ):
            provider.complete("sys", [], [], on_delta=lambda _: None)

    def test_streaming_stops_promptly_when_cancel_event_is_set(self):
        """Regression: cancel_event was accepted nowhere in the streaming
        read loop, so pressing Esc/Cancel mid-response did nothing until
        the provider finished the turn on its own."""
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        chunks = [
            {"choices": [{"delta": {"content": "first"}}]},
            {"choices": [{"delta": {"content": "second"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        sse_body = (
            "".join(f"data: {json.dumps(c)}\n" for c in chunks)
            + "data: [DONE]\n"
        ).encode()

        cancel_event = threading.Event()

        def on_delta(_chunk):
            cancel_event.set()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(sse_body),
        ):
            result = provider.complete(
                "sys", [], [], on_delta=on_delta, cancel_event=cancel_event
            )

        self.assertEqual(result.text, "first")
        self.assertEqual(result.stop_reason, "cancelled")

    def test_streaming_requests_usage_and_captures_it_from_final_chunk(self):
        """Regression: streaming previously sent `stream: True` with no
        `stream_options.include_usage`, so OpenAI-compatible servers that
        only report usage when asked (e.g. Ollama's /v1/chat/completions)
        never sent a usage chunk and every streamed response reported
        0/0 tokens."""
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        chunks = [
            {"choices": [{"delta": {"content": "hi"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        ]
        sse_body = (
            "".join(f"data: {json.dumps(c)}\n" for c in chunks)
            + "data: [DONE]\n"
        ).encode()

        captured_requests = []

        def fake_urlopen(request, timeout=None):
            captured_requests.append(json.loads(request.data))
            return _FakeResponse(sse_body)

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = provider.complete("sys", [], [], on_delta=lambda _: None)

        self.assertEqual(
            captured_requests[0]["stream_options"], {"include_usage": True}
        )
        self.assertEqual(result.usage.input_tokens, 12)
        self.assertEqual(result.usage.output_tokens, 3)

    def test_streaming_merges_include_usage_into_callers_stream_options(
        self,
    ):
        """A caller opting out via params.stream_options.include_usage
        (e.g. a provider that rejects the field) must not be silently
        overwritten, and sibling flags must survive the merge."""
        path = self._write_config(
            {
                "url": "http://fake/v1/chat/completions",
                "model": "my-model",
                "request": {
                    "params": {
                        "stream_options": {
                            "include_usage": False,
                            "include_intermediary_tokens": True,
                        }
                    }
                },
            }
        )
        provider = GenericProvider.from_json(path)

        sse_body = (
            b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n'
            b"data: [DONE]\n"
        )

        captured_requests = []

        def fake_urlopen(request, timeout=None):
            captured_requests.append(json.loads(request.data))
            return _FakeResponse(sse_body)

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.complete("sys", [], [], on_delta=lambda _: None)

        self.assertEqual(
            captured_requests[0]["stream_options"],
            {"include_usage": False, "include_intermediary_tokens": True},
        )

    def test_include_usage_in_stream_false_omits_stream_options(self):
        """Strict OpenAI-compatible endpoints that 400 on unknown fields
        can opt out entirely via config."""
        path = self._write_config(
            {
                "url": "http://fake/v1/chat/completions",
                "model": "my-model",
                "include_usage_in_stream": False,
            }
        )
        provider = GenericProvider.from_json(path)

        sse_body = (
            b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n'
            b"data: [DONE]\n"
        )

        captured_requests = []

        def fake_urlopen(request, timeout=None):
            captured_requests.append(json.loads(request.data))
            return _FakeResponse(sse_body)

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.complete("sys", [], [], on_delta=lambda _: None)

        self.assertNotIn("stream_options", captured_requests[0])

    def test_streaming_cuts_a_looping_response_short(self):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        block = "The quick brown fox jumps over. "
        chunks = [
            {"choices": [{"delta": {"content": ch}}]} for ch in block * 6
        ]
        chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        sse_body = (
            "".join(f"data: {json.dumps(c)}\n" for c in chunks)
            + "data: [DONE]\n"
        ).encode()

        deltas = []
        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(sse_body),
        ):
            result = provider.complete("sys", [], [], on_delta=deltas.append)

        self.assertEqual(result.stop_reason, "repetition")
        self.assertLess(len(result.text), len(block * 6))
        self.assertEqual("".join(deltas), result.text)

    def test_streaming_does_not_flag_normal_prose_as_repetition(self):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        prose = (
            "This is an ordinary, non-repeating explanation of the change "
            "that goes on for a while without ever looping back on itself, "
            "so it should stream through untouched by the repetition guard."
        )
        chunks = [{"choices": [{"delta": {"content": prose}}]}]
        chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        sse_body = (
            "".join(f"data: {json.dumps(c)}\n" for c in chunks)
            + "data: [DONE]\n"
        ).encode()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(sse_body),
        ):
            result = provider.complete("sys", [], [], on_delta=lambda _: None)

        self.assertEqual(result.text, prose)
        self.assertEqual(result.stop_reason, "stop")

    def test_repetition_thresholds_are_configurable_per_provider(self):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)
        provider.repetition_min_period = 1
        provider.repetition_max_period = 4
        provider.repetition_repeats = 3

        short_loop = "ab" * 6
        chunks = [
            {"choices": [{"delta": {"content": ch}}]} for ch in short_loop
        ]
        chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        sse_body = (
            "".join(f"data: {json.dumps(c)}\n" for c in chunks)
            + "data: [DONE]\n"
        ).encode()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(sse_body),
        ):
            result = provider.complete("sys", [], [], on_delta=lambda _: None)

        self.assertEqual(result.stop_reason, "repetition")
        self.assertLess(len(result.text), len(short_loop))

    def test_streaming_falls_back_to_fenced_tool_call_when_narrated_as_prose(
        self,
    ):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        narrated = (
            'Sure, I will read that file:\n```json\n{"tool": "read_file", '
            '"arguments": {"path": "a.py"}}\n```\n'
        )
        chunks = [
            {"choices": [{"delta": {"content": narrated}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        sse_body = (
            "".join(f"data: {json.dumps(c)}\n" for c in chunks)
            + "data: [DONE]\n"
        ).encode()

        deltas = []
        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(sse_body),
        ):
            result = provider.complete(
                "sys",
                [],
                [{"name": "read_file", "description": "", "parameters": {}}],
                on_delta=deltas.append,
            )

        self.assertEqual(result.text, narrated)
        self.assertEqual("".join(deltas), result.text)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "a.py"})

    def test_streaming_fenced_tool_calls_get_unique_ids_across_turns(self):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)
        tools = [{"name": "read_file", "description": "", "parameters": {}}]

        def _sse(narrated: str) -> bytes:
            chunks = [
                {"choices": [{"delta": {"content": narrated}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            return (
                "".join(f"data: {json.dumps(c)}\n" for c in chunks)
                + "data: [DONE]\n"
            ).encode()

        narrated = '```json\n{"tool": "read_file", "arguments": {"path": "a.py"}}\n```'

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            side_effect=[
                _FakeResponse(_sse(narrated)),
                _FakeResponse(_sse(narrated)),
            ],
        ):
            first = provider.complete(
                "sys", [], tools, on_delta=lambda _: None
            )
            second = provider.complete(
                "sys", [], tools, on_delta=lambda _: None
            )

        self.assertNotEqual(first.tool_calls[0].id, second.tool_calls[0].id)

    def test_streaming_ignores_fenced_json_naming_an_unknown_tool(self):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )
        provider = GenericProvider.from_json(path)

        narrated = (
            '```json\n{"tool": "delete_universe", "arguments": {}}\n```\n'
        )
        chunks = [
            {"choices": [{"delta": {"content": narrated}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        sse_body = (
            "".join(f"data: {json.dumps(c)}\n" for c in chunks)
            + "data: [DONE]\n"
        ).encode()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(sse_body),
        ):
            result = provider.complete(
                "sys",
                [],
                [{"name": "read_file", "description": "", "parameters": {}}],
                on_delta=lambda _: None,
            )

        self.assertEqual(result.text, narrated)
        self.assertEqual(result.tool_calls, [])

    def test_custom_response_paths(self):
        path = self._write_config(
            {
                "url": "http://fake/answer",
                "model": "my-model",
                "response_paths": {
                    "text": "result.answer",
                    "input_tokens": "meta.tokens_in",
                    "output_tokens": "meta.tokens_out",
                },
            }
        )
        provider = GenericProvider.from_json(path)

        response_body = json.dumps(
            {
                "result": {"answer": "custom shape works"},
                "meta": {"tokens_in": 7, "tokens_out": 4},
            }
        ).encode()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(response_body),
        ):
            result = provider.complete("sys", [], [])

        self.assertEqual(result.text, "custom shape works")
        self.assertEqual(result.usage.input_tokens, 7)
        self.assertEqual(result.usage.output_tokens, 4)

    def test_custom_tool_call_paths(self):
        path = self._write_config(
            {
                "url": "http://fake/answer",
                "model": "my-model",
                "response_paths": {
                    "text": "text",
                    "tool_calls": "actions",
                    "tool_call_id": "call_id",
                    "tool_call_name": "fn",
                    "tool_call_arguments": "args",
                },
            }
        )
        provider = GenericProvider.from_json(path)

        response_body = json.dumps(
            {
                "text": "",
                "actions": [
                    {
                        "call_id": "1",
                        "fn": "read_file",
                        "args": '{"path": "a.py"}',
                    }
                ],
            }
        ).encode()

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(response_body),
        ):
            result = provider.complete("sys", [], [])

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "a.py"})

    def test_response_paths_config_still_streams(self):
        path = self._write_config(
            {
                "url": "http://fake/answer",
                "model": "my-model",
                "response_paths": {"text": "result.answer"},
            }
        )
        provider = GenericProvider.from_json(path)

        chunks = [
            {"choices": [{"delta": {"content": "hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        sse_body = (
            "".join(f"data: {json.dumps(c)}\n" for c in chunks)
            + "data: [DONE]\n"
        ).encode()

        deltas = []
        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(sse_body),
        ):
            result = provider.complete("sys", [], [], on_delta=deltas.append)

        self.assertEqual(deltas, ["hel", "lo"])
        self.assertEqual(result.text, "hello")

    def test_anthropic_response_shape_falls_back_to_a_single_on_delta_call(
        self,
    ):
        path = self._write_config(
            {
                "url": "http://fake/answer",
                "model": "my-model",
                "response_shape": "anthropic",
            }
        )
        provider = GenericProvider.from_json(path)

        response_body = json.dumps(
            {
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }
        ).encode()

        deltas = []
        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            return_value=_FakeResponse(response_body),
        ):
            result = provider.complete("sys", [], [], on_delta=deltas.append)

        self.assertEqual(deltas, ["hello"])
        self.assertEqual(result.text, "hello")


class TestGetProviderJsonDispatch(GenericProviderTestCase):
    def test_get_provider_loads_json_config_by_path(self):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "my-model"}
        )

        provider = get_provider(str(path))

        self.assertIsInstance(provider, GenericProvider)
        self.assertEqual(provider.model, "my-model")

    def test_get_provider_model_kwarg_overrides_json_config(self):
        path = self._write_config(
            {"url": "http://fake/v1/chat/completions", "model": "from-json"}
        )

        provider = get_provider(str(path), model="from-cli")

        self.assertEqual(provider.model, "from-cli")


class TestReloadThreadSafety(GenericProviderTestCase):
    def test_concurrent_complete_never_sees_a_mixed_config(self):
        path = self._write_config({"url": "http://fake/A", "model": "model-A"})
        provider = GenericProvider.from_json(path)

        seen: list[tuple[str, str]] = []
        seen_lock = threading.Lock()

        def fake_urlopen(request, timeout=None):
            body = json.loads(request.data)
            with seen_lock:
                seen.append((request.full_url, body["model"]))
            payload = json.dumps(
                {
                    "choices": [
                        {"message": {"content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {},
                }
            ).encode()
            return _FakeResponse(payload)

        def flip_config():
            for i in range(50):
                tag = "A" if i % 2 == 0 else "B"
                path.write_text(
                    json.dumps(
                        {"url": f"http://fake/{tag}", "model": f"model-{tag}"}
                    )
                )
                provider.reload()

        def call_complete():
            for _ in range(50):
                provider.complete("sys", [], [])

        with mock.patch(
            "pycodeloop.providers.generic.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            reloader = threading.Thread(target=flip_config)
            caller = threading.Thread(target=call_complete)
            reloader.start()
            caller.start()
            reloader.join()
            caller.join()

        self.assertTrue(seen)
        for url, model in seen:
            tag = url.rsplit("/", 1)[-1]
            self.assertEqual(model, f"model-{tag}")


if __name__ == "__main__":
    unittest.main()
