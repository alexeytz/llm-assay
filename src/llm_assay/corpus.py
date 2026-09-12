import os
import time
import hashlib
import requests
from typing import Any, Optional
from urllib.parse import urlparse, unquote

# Gutenberg (and many CDNs) answer python-requests' default agent with 403/503
# while serving an ordinary browser fine, so identify the tool properly.
_USER_AGENT = "llm-assay/corpus (+https://github.com/alexeytz/llm-assay)"
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 30
_RETRIES = 3


def _local_path(book_url: str) -> Optional[str]:
    """Return a filesystem path if ``book_url`` names one, else None.

    Accepts a plain path or a file:// URL. Being able to point at a local file
    is what makes an offline or hermetic run possible at all -- the corpus is
    just a source of realistic prompt text, and nothing about it needs to come
    from the internet.
    """
    if book_url.startswith("file://"):
        return unquote(urlparse(book_url).path)
    if "://" in book_url:
        return None
    return book_url if os.path.isfile(book_url) else None


def _download(url: str) -> str:
    """Fetch the corpus, retrying transient failures.

    One flaky fetch used to abort the whole benchmark: no timeout (so a black-
    holed connection hung forever), no retry, and no User-Agent.
    """
    last: Optional[Exception] = None
    for attempt in range(1, _RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            response.raise_for_status()
            if not response.text.strip():
                raise ValueError("server returned an empty body")
            return response.text
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised at the end
            last = exc
            if attempt < _RETRIES:
                delay = 2 ** (attempt - 1)
                print(f"  fetch failed ({exc}); retrying in {delay}s "
                      f"[attempt {attempt + 1} of {_RETRIES}]")
                time.sleep(delay)
    raise RuntimeError(f"could not download {url} after {_RETRIES} attempts: {last}")


class LightweightTokenizer:
    # Set by load_tokenizer's last-resort branch to the name that failed to
    # load, so a caller counting tokens for a depth can tell the count came
    # from gpt2 rather than the model's real vocabulary. None means no
    # fallback happened.
    llm_assay_tokenizer_fallback: Optional[str] = None

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, name: str) -> "LightweightTokenizer":
        from tokenizers import Tokenizer

        if os.path.isfile(name):
            tokenizer = Tokenizer.from_file(name)
        else:
            tokenizer_json = os.path.join(name, "tokenizer.json")
            if os.path.isdir(name) and os.path.exists(tokenizer_json):
                tokenizer = Tokenizer.from_file(tokenizer_json)
            else:
                tokenizer = Tokenizer.from_pretrained(name)
        return cls(tokenizer)

    def encode(self, text: str, add_special_tokens: bool = False):
        return self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).ids

    def decode(self, token_ids):
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _transformers_tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        name,
        use_fast=True,
        trust_remote_code=False,
    )


def load_tokenizer(model_name: str, tokenizer_name: Optional[str] = None) -> Any:
    """The tokenizer alone, with the same fallback chain and no corpus.

    Split out because `probe` generates its own text and needs only a token
    counter: loading a three-megabyte novel to count tokens is pure overhead,
    and requiring a --book-url for a run that never reads one invites the reader
    to think the book mattered to the result.

    The fallback is deliberately loud, not silent -- gpt2 counts a different
    number of tokens for the same text, so every size downstream shifts with it.
    """
    name = tokenizer_name if tokenizer_name else model_name
    lightweight_error = None

    try:
        return LightweightTokenizer.from_pretrained(name)
    except Exception as e:
        lightweight_error = e

    try:
        return _transformers_tokenizer(name)
    except Exception as e:
        print(
            f"Error loading tokenizer '{name}': {e} "
            f"(lightweight tokenizer error: {lightweight_error})"
        )
        print("Falling back to 'gpt2' tokenizer as approximation.")
        # Tagged on the object rather than returned separately, so every
        # existing caller keeps working unchanged and only one that checks
        # for the tag learns anything happened. Without this a caller that
        # counts tokens for a depth (the probe) has no way to know the count
        # came from the wrong vocabulary -- gpt2 and a model's real tokenizer
        # do not agree on how many tokens a given text is, so every depth
        # downstream silently shifts and the run looks like it measured the
        # depth it was asked for when it did not.
        try:
            tokenizer = LightweightTokenizer.from_pretrained("gpt2")
        except Exception:
            tokenizer = _transformers_tokenizer("gpt2")
        try:
            tokenizer.llm_assay_tokenizer_fallback = name
        except Exception:
            pass
        return tokenizer


def tokenizer_fallback(tokenizer: Any) -> Optional[str]:
    """The tokenizer name that failed to load, or None if none did.

    The tag is set on the object rather than returned, so every caller that
    does not care stays unchanged. Reading it through one function keeps the
    getattr default in a single place: spelled out at each call site, a typo
    in the attribute name reads as 'nothing fell back' and restores exactly
    the silence the tag exists to break.
    """
    name = getattr(tokenizer, "llm_assay_tokenizer_fallback", None)
    return name if isinstance(name, str) and name else None


class TokenizedCorpus:
    def __init__(self, book_url: str, tokenizer_name: Optional[str], model_name: str):
        self.book_url = book_url
        self.tokenizer = self._get_tokenizer(model_name, tokenizer_name)
        self.tokens = self._load_data()

    def _get_transformers_tokenizer(self, name: str):
        return _transformers_tokenizer(name)

    def _get_tokenizer(self, model_name: str, tokenizer_name: Optional[str] = None):
        return load_tokenizer(model_name, tokenizer_name)

    def _load_data(self):
        try:
            # Create cache directory if it doesn't exist. Renamed with the tool:
            # the only install was the developer's, so "stranding existing
            # users' corpora" was a cost with nobody to pay it. Renaming it
            # later would not be free -- each install re-fetches a 3MB novel to
            # arrive at bytes it already has.
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "llm-assay")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Generate hash of the URL for the filename
            url_hash = hashlib.md5(self.book_url.encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{url_hash}.txt")
            
            local = _local_path(self.book_url)
            if local is not None:
                # A local corpus is the escape hatch when the network is down,
                # the host is offline, or a test needs to be hermetic. Not
                # cached: reading it again is cheaper than the cache lookup.
                print(f"Loading text from file: {local}")
                with open(local, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                if not text.strip():
                    raise ValueError(f"corpus file is empty: {local}")
            elif os.path.exists(cache_file):
                print(f"Loading text from cache: {cache_file}")
                with open(cache_file, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                print(f"Downloading book from {self.book_url}...")
                text = _download(self.book_url)
                # Basic cleanup
                start_idx = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
                if start_idx != -1:
                    text = text[start_idx:]

                # Save to cache
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved text to cache: {cache_file}")

            return self._tokenize(text)
        except Exception as e:
            print(f"Error reading or processing the corpus: {e}")
            print(
                "  The corpus is only a source of realistic prompt text. If the network\n"
                "  is unavailable, pass a local file instead:\n"
                "      --book-url /path/to/any-large-text.txt"
            )
            exit(1)

    _PG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"

    def _tokenize(self, text: str) -> Any:
        """Tokenize the corpus, dropping any Project Gutenberg licence tail.

        The licence block after the END marker is ~3% of the default book and is
        not prose: it is formulaic legal boilerplate, which a speculative draft
        head predicts far better than Conan Doyle. A slice landing in it measures
        the boilerplate rather than the model.

        Dropping it outright would move every corpus offset, because the offset
        is derived modulo ``pool_len - total_needed`` -- so every result saved
        before this would stop pairing with every result saved after, silently.
        The licence is a *suffix*, though, so repeating the story to fill the gap
        keeps the pool exactly as long as it was: every index below the marker
        is untouched, ``max_start`` is unchanged, and only the slices that used
        to read boilerplate now read prose. Those are precisely the ones worth
        changing.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        end = text.find(self._PG_END)
        if end == -1:
            return tokens

        story = self.tokenizer.encode(text[:end], add_special_tokens=False)
        if not story or len(story) >= len(tokens):
            return tokens

        # Cycle the story to restore the original length exactly.
        pad = len(tokens) - len(story)
        repeats = pad // len(story) + 1
        padded = list(story) + (list(story) * repeats)[:pad]
        print(
            f"Dropped {pad} tokens of Project Gutenberg licence text "
            f"(corpus length unchanged at {len(padded)}, so saved results still pair)"
        )
        return padded

    def get_tokenizer(self):
        return self.tokenizer

    def get_tokens(self):
        return self.tokens

    def __len__(self):
        return len(self.tokens)
