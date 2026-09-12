import hashlib
import uuid
import numpy as np
from typing import Any, List, Optional, Tuple

from .corpus import TokenizedCorpus


class PromptGenerator:
    """Builds (context, prompt) pairs sampled from the corpus.

    Sampling is random by default. Passing ``seed`` makes it deterministic *and
    keyed*: the corpus offset is derived from ``(seed, key)`` rather than from a
    running PRNG stream. That matters for A/B runs -- two configs see identical
    text for the same logical slot even if they issue a different number of
    warmup or latency probes in between, so a paired comparison stays valid and
    content-driven variance cancels out.
    """

    def __init__(
        self,
        corpus: TokenizedCorpus,
        seed: Optional[int] = None,
        run_salt: Optional[str] = None,
    ):
        self.corpus = corpus
        # Set by --cold. Mixing it into the offset derivation moves every slice
        # of the corpus, so a repeat invocation cannot share a prefix with the
        # last one. A cache-buster appended to the prompt cannot achieve this:
        # prefix caching matches on the *prefix*, and a suffix leaves it intact.
        self.run_salt = run_salt
        self.tokenizer = corpus.get_tokenizer()
        self.all_tokens = corpus.get_tokens()
        self.seed = seed

    def _derive(self, key: Any, salt: str, modulo: int) -> int:
        # Byte-compatible with earlier versions when no salt is in play, so a
        # result saved before --cache-buster existed still pairs with a new one.
        parts = (self.seed, salt, key) if self.run_salt is None else (
            self.seed, self.run_salt, salt, key
        )
        payload = repr(parts).encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "big") % modulo

    def _start_index(self, pool_len: int, total_needed: int, key: Any) -> int:
        max_start = pool_len - total_needed
        if max_start <= 0:
            return 0
        if self.seed is None:
            return int(np.random.randint(0, max_start))
        return self._derive(key, "offset", max_start)

    def _cache_buster(self, key: Any) -> str:
        if self.seed is None:
            return f" {uuid.uuid4()}"
        # Deterministic but unique per slot, so --no-cache stays reproducible.
        payload = repr((self.seed, "nocache", key)).encode()
        return f" {uuid.UUID(bytes=hashlib.blake2b(payload, digest_size=16).digest())}"

    def generate(
        self,
        prompt_tokens: int,
        context_tokens: int = 0,
        no_cache: bool = False,
        key: Any = None,
    ) -> Tuple[str, str]:
        """
        Generates a single (context, prompt) pair.
        """
        suffix = ""
        suffix_len = 0
        if no_cache:
            suffix = self._cache_buster(key)
            suffix_len = len(self.tokenizer.encode(suffix, add_special_tokens=False))

        # Adjust prompt tokens to fetch from text
        text_prompt_tokens = max(0, prompt_tokens - suffix_len)

        # Create a pool of tokens large enough
        total_needed = text_prompt_tokens + context_tokens

        # Create a local reference to tokens to potentially extend
        current_tokens = self.all_tokens

        if len(current_tokens) < total_needed:
            # Repeat tokens if not enough
            current_tokens = current_tokens * (total_needed // len(current_tokens) + 2)

        start_idx = self._start_index(len(current_tokens), total_needed, key)

        selected_tokens = current_tokens[start_idx : start_idx + total_needed]

        context_text = (
            self.tokenizer.decode(selected_tokens[:context_tokens])
            if context_tokens > 0
            else ""
        )
        prompt_text = self.tokenizer.decode(selected_tokens[context_tokens:])

        if no_cache:
            prompt_text += suffix

        return context_text, prompt_text

    def generate_batch(
        self,
        batch_size: int,
        prompt_tokens: int,
        context_tokens: int = 0,
        no_cache: bool = False,
        key: Any = None,
    ) -> List[Tuple[str, str]]:
        """
        Generates a batch of (context, prompt) pairs.
        """
        return [
            self.generate(prompt_tokens, context_tokens, no_cache, key=(key, i))
            for i in range(batch_size)
        ]
