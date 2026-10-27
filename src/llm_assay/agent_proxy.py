"""A recording pass-through between the harness and the endpoint.

Three times the harness sent something other than what the arm asked for, and
each was found by inference after the arm had run: Pi dropped the thinking
level (so every arm thought at its deployment's default, and one ran unthinking
while its file said `medium`), sent nothing for `off`, and clamped `xhigh` to
"high", which the server rejected 156 times. The request is the record, so
this records it, and checks it.

Each run gets its own path prefix, ``/r/<run>/...``, so every request is
attributed to the run that sent it whatever else is in flight. The proxy:

* checks every request body against what the harness says the arm's settings
  must look like on the wire (``Harness.expected_request``). A mismatch is
  answered 400 without reaching the endpoint, and recorded; the arm voids the
  run and stops, rather than measure a setting nobody chose;
* records each request's parameters (never its headers, which carry the key)
  and each response's ``finish_reason`` and usage;
* reassembles streamed tool calls and counts the ones the model got wrong in
  structure: arguments that are not a JSON object, or a tool the request did
  not offer. That is a fact about the model's output a pass/fail cannot show.
  A call the server's tool parser could not recognise at all reaches us as
  plain text and is not counted -- this is a floor, not the rate;
* with ``--keep``, writes every request body and its response summary to
  ``requests.jsonl`` in the run directory: the exact prefixes a next-token
  comparison needs.

Nothing is buffered on the way back: chunks are written as they arrive. A
client that disconnects is only noticed at the next chunk, and a request queued
or prefilling sends none, so a timed-out run's last request used to stay on the
server -- load on every run after it, which the next arm's start reported as
"3 request(s) already running". Releasing a run cancels whatever it still has
in flight, which closes the upstream request at once.
"""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

#: The request fields worth recording: what decides how the model samples and
#: thinks. Messages and tools are counted, not copied, outside requests.jsonl.
PARAMS = ("model", "stream", "max_tokens", "max_completion_tokens", "temperature", "top_p",
          "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty",
          "seed", "reasoning_effort", "chat_template_kwargs", "tool_choice", "n")

#: In an expected-request mapping: the field must not be sent at all.
ABSENT = object()

#: Request headers not forwarded: hop-by-hop, or set by the client library.
#: Accept-Encoding is dropped so the stream arrives uncompressed and readable.
_DROP = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding",
         "keep-alive"}


def mismatches(body: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    """Each expected field the body does not carry, as a sentence.

    A dotted key reaches into a nested mapping: ``chat_template_kwargs.enable_thinking``.
    """
    out = []
    for key, want in expected.items():
        node: Any = body
        found = True
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                found = False
                break
        if want is ABSENT:
            if found:
                out.append(f"{key} was sent as {node!r}; it should be absent")
        elif not found:
            out.append(f"{key} was not sent; it should be {want!r}")
        elif node != want:
            out.append(f"{key} was sent as {node!r}; it should be {want!r}")
    return out


@dataclass
class Recorder:
    """What one run's harness sent and got back."""

    expected: Dict[str, Any]
    keep: Optional[str] = None
    requests: int = 0
    first_params: Optional[Dict[str, Any]] = None
    mismatch: List[str] = field(default_factory=list)
    finish_reasons: Dict[str, int] = field(default_factory=dict)
    statuses: Dict[str, int] = field(default_factory=dict)
    tool_calls: int = 0
    malformed: List[Dict[str, str]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    inflight: Set["asyncio.Task[Any]"] = field(default_factory=set)

    def summary(self) -> Dict[str, Any]:
        return {"count": self.requests, "params": self.first_params,
                "mismatch": self.mismatch, "statuses": self.statuses,
                "finish_reasons": self.finish_reasons, "tool_calls": self.tool_calls,
                "tool_calls_malformed": len(self.malformed),
                "malformed_examples": self.malformed[:5],
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}

    def _log(self, entry: Dict[str, Any]) -> None:
        if self.keep:
            with open(self.keep, "a") as fh:
                fh.write(json.dumps(entry) + "\n")


class _Stream:
    """Reads an SSE chat-completions stream as it passes, without holding it."""

    def __init__(self, offered: List[str]):
        self.offered = offered
        self.pending = b""
        self.calls: Dict[int, Dict[str, str]] = {}
        self.finish: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None

    def feed(self, chunk: bytes) -> None:
        self.pending += chunk
        *lines, self.pending = self.pending.split(b"\n")
        for line in lines:
            self._line(line.strip())

    def close(self) -> None:
        if self.pending.strip():
            self._line(self.pending.strip())
        self.pending = b""

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            return
        try:
            event = json.loads(data)
        except ValueError:
            return
        self.event(event)

    def event(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                self.finish = str(choice["finish_reason"])
            part = choice.get("delta") or choice.get("message") or {}
            calls = part.get("tool_calls") if isinstance(part, dict) else None
            for call in calls or []:
                if not isinstance(call, dict):
                    continue
                slot = self.calls.setdefault(int(call.get("index", len(self.calls))),
                                             {"name": "", "arguments": ""})
                fn = call.get("function") or {}
                slot["name"] += fn.get("name") or ""
                args = fn.get("arguments")
                if isinstance(args, str):
                    slot["arguments"] += args
                elif args is not None:
                    slot["arguments"] += json.dumps(args)

    def malformed(self) -> List[Dict[str, str]]:
        bad = []
        for call in self.calls.values():
            try:
                ok = isinstance(json.loads(call["arguments"]), dict)
            except ValueError:
                ok = False
            if not ok:
                bad.append({"tool": call["name"], "why": "arguments are not a JSON object",
                            "arguments": call["arguments"][:300]})
            elif self.offered and call["name"] not in self.offered:
                bad.append({"tool": call["name"], "why": "the request offered no such tool",
                            "arguments": call["arguments"][:300]})
        return bad


class Proxy:
    """One per arm; started inside the arm's event loop, on localhost."""

    def __init__(self, upstream: str):
        parts = urlsplit(upstream.rstrip("/"))
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self.path = parts.path
        self.runs: Dict[str, Recorder] = {}
        self._runner: Optional[web.AppRunner] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self.port = 0

    async def start(self) -> None:
        app = web.Application(client_max_size=1 << 30)
        app.router.add_route("*", "/r/{run}/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        await web.SockSite(self._runner, sock).start()
        # No total timeout: a run is bounded by its own --timeout, which kills
        # the harness, and that disconnect is what closes the upstream request.
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30), trust_env=True,
            auto_decompress=False)

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
        if self._runner:
            await self._runner.cleanup()

    def base_url(self, run: str) -> str:
        """What the harness is told the endpoint is, for this run."""
        return f"http://127.0.0.1:{self.port}/r/{run}{self.path}"

    def register(self, run: str, expected: Dict[str, Any], keep: Optional[str] = None) -> Recorder:
        self.runs[run] = Recorder(expected=expected, keep=keep)
        return self.runs[run]

    async def release(self, run: str) -> Optional[Recorder]:
        """End a run: cancel what it still has in flight, and wait for that to be logged."""
        rec = self.runs.pop(run, None)
        if rec and rec.inflight:
            for task in rec.inflight:
                task.cancel()
            await asyncio.gather(*rec.inflight, return_exceptions=True)
        return rec

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        rec = self.runs.get(request.match_info["run"])
        if rec is None:
            return web.json_response({"error": {"message": "llm-assay proxy: unknown run"}},
                                     status=404)
        task = asyncio.current_task()
        assert task is not None
        rec.inflight.add(task)
        try:
            return await self._forward(request, rec)
        finally:
            rec.inflight.discard(task)

    async def _forward(self, request: web.Request, rec: Recorder) -> web.StreamResponse:
        raw = await request.read()
        body: Any = None
        if raw:
            try:
                body = json.loads(raw)
            except ValueError:
                body = None
        is_generation = isinstance(body, dict) and "messages" in body
        offered: List[str] = []
        entry: Dict[str, Any] = {"path": "/" + request.match_info["tail"]}
        if is_generation:
            rec.requests += 1
            params = {k: body[k] for k in PARAMS if k in body}
            if rec.first_params is None:
                rec.first_params = params
            offered = [t.get("function", {}).get("name", "") for t in body.get("tools") or []
                       if isinstance(t, dict)]
            entry.update(params=params, messages=body["messages"], tools=offered)
            wrong = mismatches(body, rec.expected)
            if wrong:
                rec.mismatch.extend(w for w in wrong if w not in rec.mismatch)
                entry["rejected"] = wrong
                rec._log(entry)
                return web.json_response(
                    {"error": {"message": "llm-assay: the request does not match the arm's "
                               "settings: " + "; ".join(wrong), "type": "invalid_request_error"}},
                    status=400)

        headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP}
        url = self.origin + "/" + request.match_info["tail"]
        if request.query_string:
            url += "?" + request.query_string
        assert self._session is not None
        stream = _Stream(offered)
        response: Optional[web.StreamResponse] = None
        whole = b""
        try:
            async with self._session.request(request.method, url, data=raw or None,
                                             headers=headers) as upstream:
                key = str(upstream.status)
                if is_generation:
                    rec.statuses[key] = rec.statuses.get(key, 0) + 1
                response = web.StreamResponse(status=upstream.status)
                for name in ("Content-Type", "Content-Encoding"):
                    if name in upstream.headers:
                        response.headers[name] = upstream.headers[name]
                await response.prepare(request)
                streaming = "event-stream" in upstream.headers.get("Content-Type", "")
                async for chunk in upstream.content.iter_any():
                    if is_generation:
                        if streaming:
                            stream.feed(chunk)
                        else:
                            whole += chunk
                    await response.write(chunk)
                await response.write_eof()
        except (ConnectionResetError, aiohttp.ClientConnectionError, OSError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
        except asyncio.CancelledError:
            entry["error"] = "cancelled: the run ended with this request in flight"
            raise
        finally:
            if is_generation:
                if whole:
                    try:
                        stream.event(json.loads(whole))
                    except ValueError:
                        pass
                stream.close()
                self._account(rec, stream, entry)
        if response is None:
            return web.json_response({"error": {"message": "llm-assay proxy: upstream "
                                                f"unreachable: {entry.get('error')}"}}, status=502)
        return response

    @staticmethod
    def _account(rec: Recorder, stream: _Stream, entry: Dict[str, Any]) -> None:
        if stream.finish:
            rec.finish_reasons[stream.finish] = rec.finish_reasons.get(stream.finish, 0) + 1
        rec.tool_calls += len(stream.calls)
        bad = stream.malformed()
        rec.malformed.extend(bad)
        if stream.usage:
            rec.prompt_tokens += int(stream.usage.get("prompt_tokens") or 0)
            rec.completion_tokens += int(stream.usage.get("completion_tokens") or 0)
        entry.update(finish_reason=stream.finish, usage=stream.usage,
                     tool_calls=list(stream.calls.values()), malformed=bad)
        rec._log(entry)
