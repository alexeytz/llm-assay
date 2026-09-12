import time
import json
import codecs
import aiohttp
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

# One flag for three different warnings meant the first to fire silenced the
# other two for the whole process -- so a run could hide the fact that it also
# fell back to local tokenization, or to one-token-per-chunk timing.
_warned_messages: set = set()

# vLLM's Rust frontend rejects empty user content, so context preloads use a
# tiny probe turn instead.
CONTEXT_LOAD_USER_MESSAGE = "."


def _warn_once(message: str):
    """Print a warning the first time this exact message occurs."""
    if message not in _warned_messages:
        print(message)
        _warned_messages.add(message)

@dataclass
class RequestResult:
    start_ts: float = 0.0
    end_ts: float = 0.0
    first_token_ts: Optional[float] = None
    first_response_ts: Optional[float] = None
    prompt_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
    token_timestamps: List[float] = field(default_factory=list)

def _answer_text(choice: Dict[str, Any]) -> Tuple[str, str]:
    """Pull (content, reasoning) out of one choice, whichever route produced it.

    /chat/completions answers under message.content, /completions under text.
    Reading only the first made the coherence check fail on the raw route for a
    model that had answered perfectly. Thinking models may leave content null
    and put everything in reasoning_content, so both are returned and the caller
    decides which it wants.
    """
    message = choice.get("message", {}) or {}
    content = message.get("content") or choice.get("text") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    return content, reasoning


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        extra_body: Optional[Dict[str, Any]] = None,
        exact_tg: bool = False,
        endpoint: str = "chat",
    ):
        # A trailing slash makes the join below produce "/v1//chat/completions",
        # which llama.cpp answers with a bare 404 "File Not Found" -- nothing in
        # the message says the URL was doubled, and the warmup dies before a
        # single shape runs. Normalise once here rather than at each leaf.
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.extra_body = extra_body or {}
        self.exact_tg = exact_tg
        # "chat" drives /v1/chat/completions, "completions" the raw
        # /v1/completions. The second isolates engine cost from chat-template
        # cost and reaches stacks that never implemented the chat route.
        self.endpoint = endpoint
        self.headers = {"Authorization": f"Bearer {api_key}"}

    @property
    def _url(self) -> str:
        leaf = "chat/completions" if self.endpoint == "chat" else "completions"
        return f"{self.base_url}/{leaf}"

    def _as_prompt(self, context_text: str, prompt_text: str) -> str:
        """Flatten a context + prompt pair into one raw-completions prompt."""
        return f"{context_text}\n\n{prompt_text}" if context_text else prompt_text

    def _probe_payload(self, text: str, max_tokens: int, context: str = "") -> Dict[str, Any]:
        """Payload for the latency, coherence and warmup probes.

        These build their own bodies rather than going through
        _build_generation_payload, so they need the same chat/completions branch
        or they would all silently keep hitting the chat route.
        """
        if self.endpoint == "chat":
            messages = ([{"role": "system", "content": context}] if context else [])
            messages.append({"role": "user", "content": text})
            return {"model": self.model_name, "messages": messages, "max_tokens": max_tokens}
        return {
            "model": self.model_name,
            "prompt": self._as_prompt(context, text),
            "max_tokens": max_tokens,
        }

    def _build_generation_payload(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        no_cache: bool,
        reasoning_effort: Optional[str] = None,
        thinking: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "stream": True,
            "return_token_ids": True,
            "stream_options": {"include_usage": True},
        }

        if self.endpoint == "chat":
            payload["messages"] = messages
        else:
            # No roles and no template: the context and the prompt are one string.
            context = next((m["content"] for m in messages if m["role"] == "system"), "")
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            payload["prompt"] = self._as_prompt(context, user)

        if no_cache:
            payload["cache_prompt"] = False

        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        # Two spellings, because deployments honour different ones and both are
        # in the wild. On vLLM + Qwen3 they render byte-identical prompts, and
        # sending both together is accepted without conflict; a stack that knows
        # only one still gets the instruction. Verified against the server's
        # /v1/chat/completions/render endpoint.
        if thinking is not None:
            kwargs = dict(payload.get("chat_template_kwargs") or {})
            if thinking == "off":
                payload["reasoning_effort"] = "none"
                kwargs["enable_thinking"] = False
            else:
                kwargs["enable_thinking"] = True
            payload["chat_template_kwargs"] = kwargs

        payload.update(self.extra_body)

        if self.exact_tg:
            payload["max_tokens"] = max_tokens
            payload["min_tokens"] = max_tokens
            payload["ignore_eos"] = True

        return payload

    @staticmethod
    def _non_empty_user_content(content: str) -> str:
        if content.strip():
            return content
        return CONTEXT_LOAD_USER_MESSAGE

    @staticmethod
    def _append_observed_token_timestamps(result: RequestResult, chunk_time: float, token_count: int):
        if token_count <= 0:
            return

        if token_count == 1:
            result.token_timestamps.append(chunk_time)
            return

        last_ts = result.token_timestamps[-1] if result.token_timestamps else result.first_token_ts
        if last_ts is None:
            last_ts = result.start_ts
        time_window = chunk_time - last_ts
        for i in range(token_count):
            ts = last_ts + (time_window * (i + 1) / token_count)
            result.token_timestamps.append(ts)

    @staticmethod
    def _interpolate_token_timestamps(chunk_times: List[float], token_count: int) -> List[float]:
        if token_count <= 0 or not chunk_times:
            return []

        if token_count == 1 or len(chunk_times) == 1:
            return [chunk_times[0]] * token_count

        first_ts = chunk_times[0]
        last_ts = chunk_times[-1]
        if last_ts <= first_ts:
            return [first_ts] * token_count

        step = (last_ts - first_ts) / (token_count - 1)
        return [first_ts + (step * i) for i in range(token_count)]

    def _finalize_stream_tokens(
            self,
            result: RequestResult,
            content_chunks: List[Dict[str, Any]],
            usage_completion_tokens: Optional[int],
            tokenizer=None
        ):
        if not content_chunks:
            if usage_completion_tokens is not None:
                result.total_tokens = usage_completion_tokens
            return

        token_id_chunks = [
            chunk for chunk in content_chunks
            if isinstance(chunk.get("token_ids"), list)
        ]

        if len(token_id_chunks) == len(content_chunks):
            for chunk in content_chunks:
                token_count = len(chunk["token_ids"])
                result.total_tokens += token_count
                self._append_observed_token_timestamps(result, chunk["timestamp"], token_count)
            return

        chunk_times = [chunk["timestamp"] for chunk in content_chunks]

        if usage_completion_tokens is not None:
            _warn_once("  No complete token_ids in response, using stream usage token count")
            result.total_tokens = usage_completion_tokens
            result.token_timestamps = self._interpolate_token_timestamps(chunk_times, usage_completion_tokens)
            return

        if tokenizer is not None:
            _warn_once("  No token_ids or usage in response, using local tokenization")
            full_content = "".join(chunk["text"] for chunk in content_chunks)
            token_count = len(tokenizer.encode(full_content, add_special_tokens=False))
            result.total_tokens = token_count
            result.token_timestamps = self._interpolate_token_timestamps(chunk_times, token_count)
            return

        _warn_once("  No token_ids, usage, or tokenizer, assuming 1 token per chunk")
        result.total_tokens = len(content_chunks)
        result.token_timestamps = chunk_times

    async def measure_latency(
        self,
        session: aiohttp.ClientSession,
        mode: str = "api",
        warmup_runs: int = 1,
        measured_runs: int = 3,
    ) -> float:
        if mode == "none":
            print("Skipping latency measurement (assuming 0 ms).")
            return 0

        warmup_runs = max(0, warmup_runs)
        measured_runs = max(0, measured_runs)
        if mode == "generation" and warmup_runs > 0:
            print(
                f"Measuring latency using mode: {mode} "
                f"({warmup_runs} warmup + {measured_runs} measured probes)..."
            )
        else:
            print(f"Measuring latency using mode: {mode}...")
        latencies = []
        total_runs = measured_runs + (warmup_runs if mode == "generation" else 0)
        
        for probe_idx in range(total_runs):
            is_warmup = mode == "generation" and probe_idx < warmup_runs
            start = time.perf_counter()
            try:
                if mode == "api":
                    # Stop at the first byte, not the end of the body. README
                    # documents this as time-to-first-byte, and it is subtracted
                    # from ttfr -- itself a first-chunk measurement -- so reading
                    # the whole body oversubtracts by the body transfer time and
                    # inflates every est_ppt in the run.
                    async with session.get(f"{self.base_url}/models", headers=self.headers) as response:
                        async for _ in response.content:
                            break
                        latencies.append(time.perf_counter() - start)
                        async for _ in response.content:
                            pass
                elif mode == "generation":
                    payload = {**self._probe_payload("hello", 1), "stream": True}
                    async with session.post(self._url, json=payload, headers=self.headers) as response:
                        async for _ in response.content:
                            elapsed = time.perf_counter() - start
                            if not is_warmup:
                                latencies.append(elapsed)
                            break
                        async for _ in response.content: pass
            except Exception as e:
                print(f"Error measuring latency: {e}")
        
        if latencies:
            avg_latency = float(np.mean(latencies))
            print(f"Average latency ({mode}): {avg_latency*1000:.2f} ms")
            return avg_latency
        return 0

    async def run_coherence_test(
        self,
        session: aiohttp.ClientSession,
        prompt: Optional[str] = None,
        expect: Optional[List[str]] = None,
    ) -> bool:
        """Check the model answers sanely before spending minutes benchmarking it.

        The default question is English and expects 'Paris'. That is a fine smoke
        test for an English-instructed model and a false failure for anything
        else -- a model answering perfectly in Chinese or German would abort the
        whole suite. So the question and the accepted answers are overridable
        (``--coherence-prompt`` / ``--coherence-expect``), and passing an empty
        expectation degrades the check to "did it produce any content at all",
        which still catches a mis-served or broken model without assuming a
        language.
        """
        print("\nRunning coherence test...")
        prompt = prompt or "What is the capital of France? Please reply with one word only"
        expected = [e.lower() for e in (expect if expect is not None else ["paris"]) if e]
        payload = self._probe_payload(prompt, 100)

        try:
            async with session.post(self._url, json=payload, headers=self.headers) as response:
                response_json = await response.json()

                if 'choices' not in response_json or len(response_json['choices']) == 0:
                    print("Coherence test FAILED: No choices in response")
                    return False

                content, reasoning = _answer_text(response_json['choices'][0])
                full_content = (content + reasoning).lower()

                if not expected:
                    # Language-agnostic mode: any real content is a pass.
                    if full_content.strip():
                        print("Coherence test PASSED (content check only).")
                        return True
                    print("Coherence test FAILED: the model returned no content.")
                    return False

                if any(term in full_content for term in expected):
                    print("Coherence test PASSED.")
                    return True

                wanted = " or ".join(repr(e) for e in expected)
                print(f"Coherence test FAILED: expected {wanted}. Got: {content[:200]}...")
                print(
                    "  If the model is not English-instructed this may be a false failure. "
                    "Use\n  --coherence-prompt/--coherence-expect to suit it, "
                    "--coherence-expect '' for a\n  content-only check, or --skip-coherence."
                )
                return False
        except Exception as e:
            print(f"Coherence test FAILED with error: {e}")
            return False

    async def warmup(self, session: aiohttp.ClientSession, tokenizer=None):
        print("Warming up...")
        warmup_text = "Warmup " * 10

        delta_user = 0
        delta_context = 0

        # 1. User only
        payload_user = self._probe_payload(warmup_text, 1)

        try:
            async with session.post(self._url, json=payload_user, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Warmup failed: HTTP {response.status}: {error_text}")
                    raise SystemExit(1)
                response_json = await response.json()
                if tokenizer:
                    if 'usage' in response_json:
                        prompt_tokens = response_json['usage']['prompt_tokens']
                        local_tokens = len(tokenizer.encode(warmup_text, add_special_tokens=False))
                        delta_user = prompt_tokens - local_tokens
                        print(f"Warmup (User only) complete. Delta: {delta_user} tokens (Server: {prompt_tokens}, Local: {local_tokens})")
                    else:
                        print("Warmup (User only) complete (no usage stats found).")
                else:
                    print("Warmup complete.")

            if tokenizer:
                # 2. Context Only
                payload_sys_probe = {
                    **self._probe_payload(CONTEXT_LOAD_USER_MESSAGE, 1, context=warmup_text),
                    "max_tokens": 1
                }
                async with session.post(self._url, json=payload_sys_probe, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Warmup failed: HTTP {response.status}: {error_text}")
                        raise SystemExit(1)
                    response_json = await response.json()
                    if 'usage' in response_json:
                        prompt_tokens = response_json['usage']['prompt_tokens']
                        local_tokens = len(tokenizer.encode(warmup_text, add_special_tokens=False))
                        probe_tokens = len(tokenizer.encode(CONTEXT_LOAD_USER_MESSAGE, add_special_tokens=False))
                        delta_context = prompt_tokens - local_tokens - probe_tokens
                        print(f"Warmup (System+Probe) complete. Delta: {delta_context} tokens (Server: {prompt_tokens}, Local context: {local_tokens}, Probe: {probe_tokens})")
                    else:
                        delta_context = delta_user
        except Exception as e:
            print(f"Warmup failed: {e}")
            raise SystemExit(1)
        return delta_user, delta_context

    async def run_generation(
            self,
            session: aiohttp.ClientSession,
            context_text: str,
            prompt_text: str,
            max_tokens: int,
            no_cache: bool,
            tokenizer=None,
            progress=None,
            request_id: Optional[int] = None,
            reasoning_effort: Optional[str] = None,
            thinking: Optional[str] = None,
        ) -> RequestResult:

        messages = []
        if context_text:
            messages.append({"role": "system", "content": context_text})
        messages.append({"role": "user", "content": self._non_empty_user_content(prompt_text)})
        
        result = RequestResult()
        
        try:
            payload = self._build_generation_payload(
                messages, max_tokens, no_cache, reasoning_effort, thinking
            )
            
            result.start_ts = time.perf_counter()

            async with session.post(self._url, json=payload, headers=self.headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    result.error = f"HTTP {response.status}: {error_text}"
                    print(result.error)
                    self._emit_request_end(progress, request_id, result)
                    return result

                decoder = codecs.getincrementaldecoder("utf-8")(errors='replace')
                buffer = ""
                content_chunks: List[Dict[str, Any]] = []
                usage_completion_tokens: Optional[int] = None
                
                async for chunk_bytes in response.content.iter_any():
                    chunk_time = time.perf_counter()
                    decoded_chunk = decoder.decode(chunk_bytes, final=False)
                    buffer += decoded_chunk
                    
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        
                        if line == 'data: [DONE]' or line == 'data:[DONE]':
                            continue
                        
                        if line.startswith('data:'):
                            try:
                                json_str = line[5:].strip()
                                chunk = json.loads(json_str)

                                usage = chunk.get('usage')
                                if isinstance(usage, dict):
                                    prompt_tokens = usage.get('prompt_tokens')
                                    if isinstance(prompt_tokens, int):
                                        result.prompt_tokens = prompt_tokens
                                    completion_tokens = usage.get('completion_tokens')
                                    if isinstance(completion_tokens, int) and completion_tokens >= 0:
                                        usage_completion_tokens = completion_tokens
                                
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    if result.first_response_ts is None:
                                        result.first_response_ts = chunk_time
                                        if progress is not None and request_id is not None:
                                            try:
                                                progress.request_first_response(
                                                    request_id=request_id,
                                                    ttfr_s=chunk_time - result.start_ts,
                                                )
                                            except Exception:
                                                pass

                                    # /chat/completions streams {delta: {content}};
                                    # /completions streams {text}. Reasoning fields
                                    # only exist on the chat route.
                                    choice = chunk['choices'][0]
                                    delta = choice.get('delta', {})
                                    content = delta.get('content')
                                    if content is None and 'text' in choice:
                                        content = choice.get('text')
                                    reasoning_content = delta.get('reasoning_content')
                                    reasoning = delta.get('reasoning')

                                    if content or reasoning_content or reasoning:
                                        if result.first_token_ts is None:
                                            result.first_token_ts = chunk_time
                                            if progress is not None and request_id is not None:
                                                try:
                                                    progress.request_first_token(
                                                        request_id=request_id,
                                                        ttft_s=chunk_time - result.start_ts,
                                                    )
                                                except Exception:
                                                    pass

                                        token_ids = chunk['choices'][0].get('token_ids')
                                        text = content or reasoning_content or reasoning
                                        content_chunks.append({
                                            "text": text,
                                            "timestamp": chunk_time,
                                            "token_ids": token_ids,
                                        })

                                        if progress is not None and request_id is not None:
                                            # Best-effort per-chunk count for the live stream.
                                            # The authoritative total is reconciled in
                                            # _finalize_stream_tokens and reported by request_end.
                                            has_token_ids = isinstance(token_ids, list)
                                            chunk_count = len(token_ids) if has_token_ids else 1
                                            try:
                                                progress.tokens(
                                                    request_id=request_id,
                                                    count=chunk_count,
                                                    snippet=text or "",
                                                    estimated=not has_token_ids,
                                                )
                                            except Exception:
                                                pass
                            except json.JSONDecodeError:
                                continue

                self._finalize_stream_tokens(result, content_chunks, usage_completion_tokens, tokenizer)
                result.end_ts = time.perf_counter()

        except Exception as e:
            print(f"Error during run: {e}")
            result.error = str(e)

        self._emit_request_end(progress, request_id, result)
        return result

    @staticmethod
    def _emit_request_end(progress, request_id: Optional[int], result: "RequestResult") -> None:
        """Emit the request_end progress event for a finished request."""
        if progress is None or request_id is None:
            return
        decode_seconds = 0.0
        if result.first_token_ts is not None and result.end_ts:
            decode_seconds = max(0.0, result.end_ts - result.first_token_ts)
        try:
            progress.request_end(
                request_id=request_id,
                total_tokens=result.total_tokens,
                prompt_tokens=result.prompt_tokens,
                decode_seconds=decode_seconds,
                error=result.error or "",
            )
        except Exception:
            pass
