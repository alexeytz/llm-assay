"""
Main entry point for the llm-assay CLI.
"""

import asyncio
import datetime
import sys
import uuid
from . import __version__
from pydantic import ValidationError

from .config import BenchmarkConfig
from .corpus import TokenizedCorpus, tokenizer_fallback
from .prompts import PromptGenerator
from .client import LLMClient
from .runner import BenchmarkRunner
from .progress import ProgressEmitter

async def main_async() -> int:
    # 1. Parse Configuration
    try:
        config = BenchmarkConfig.from_args()
    except ValidationError as exc:
        # A bad flag is a usage error, not a crash: print what argparse would
        # have printed rather than a pydantic traceback.
        for err in exc.errors():
            print(f"error: {err.get('msg', '').removeprefix('Value error, ')}", file=sys.stderr)
        return 2

    # 1b. If JSONL is going to stdout, route llm-assay's status prints to
    # stderr so they don't corrupt the JSONL stream a consumer is parsing.
    if config.emit_progress == "-":
        sys.stdout = sys.stderr

    # 2. Print Header
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"llm-assay ({__version__})")
    print(f"Date: {current_time}")
    print(f"Benchmarking model: {config.model} at {config.base_url}")
    print(f"Concurrency levels: {config.concurrency_levels}")

    # 3. Prepare Data
    corpus = TokenizedCorpus(config.book_url, config.tokenizer, config.model)
    print(f"Total tokens available in text corpus: {len(corpus)}")

    # load_tokenizer falls back to gpt2 when the requested tokenizer will not
    # load -- right for a benchmark that only needs realistic text, wrong for
    # the labels: gpt2 and a model vocabulary disagree on how many tokens a
    # given text is, so every --pp and --depth shifts. Measured against a
    # GGUF-only repo, --pp 512 sent 485 tokens and the row still said pp512.
    # The rate stays honest (the server's own prompt_tokens is preferred when
    # it is within 20% of expected) -- it is the shape that is misdescribed.
    fell_back = tokenizer_fallback(corpus.get_tokenizer())
    if fell_back:
        print(
            f"[WARNING] tokenizer {fell_back!r} would not load; gpt2 counted the\n"
            "          tokens instead, so every --pp and --depth label below is an\n"
            "          approximation of the shape actually sent. Pass --tokenizer at\n"
            "          the base model when the served repo is GGUF-only."
        )

    # Past the end of the book the corpus is repeated, which is invisible in the
    # output and in the saved result. Repeated text is unusually easy for a draft
    # head to predict, so a deep run can be measuring the corpus as much as the
    # server. Say so rather than letting it pass silently.
    deepest = max(config.depths) + max(config.pp_counts)
    if deepest > len(corpus):
        print(
            f"[WARNING] the deepest shape needs {deepest} tokens but the corpus has "
            f"{len(corpus)}.\n"
            "          The text is REPEATED to fill the difference. Repetition is easy "
            "for a\n"
            "          speculative draft head to predict, so decode numbers at that depth "
            "may\n"
            "          flatter the server. Pass a longer --book-url to measure "
            "unrepeated text."
        )

    # 4. Initialize Components
    # A value unique to this process, so no two invocations draw the same slices.
    run_salt = uuid.uuid4().hex[:12] if config.cold else None
    if run_salt:
        print(f"Cold mode: corpus offsets salted with {run_salt} (prefix cache cannot be warm)")
    prompt_gen = PromptGenerator(corpus, seed=config.seed, run_salt=run_salt)
    client = LLMClient(
        config.base_url,
        config.api_key,
        config.served_model_name,
        config.extra_body,
        config.exact_tg,
        config.endpoint,
    )

    progress = None
    if config.emit_progress:
        progress = ProgressEmitter(config.emit_progress, llm_assay_version=__version__)
    runner = BenchmarkRunner(config, client, prompt_gen, progress=progress)
    runner.run_salt = run_salt

    # 5. Run Benchmark Suite
    status = "ok"
    try:
        await runner.run_suite()
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "error"
        raise
    finally:
        if progress is not None:
            try:
                progress.bench_complete(status=status)
            finally:
                progress.close()

    print(f"\nllm-assay ({__version__})")
    print(f"date: {current_time} | latency mode: {config.latency_mode}")

    # A suite that measured nothing is a failed run, not a successful empty one.
    # Every request can fail -- the server refusing an over-long context, or
    # running out of memory partway through a full-context prefill -- and until
    # now that still exited 0 unless --exit-on-first-fail happened to be set.
    # Anything automated reading the exit code was told the benchmark succeeded.
    if not runner.results.has_measurements():
        print(
            "error: no measurements were collected -- every request failed. "
            "The output above has the server's reason.",
            file=sys.stderr,
        )
        return 1

    summary = runner.results.failure_summary()
    if summary:
        print("\n" + summary)

    # A shape with nothing in it is a hole in the data, and the table prints
    # around the hole looking complete. Losing individual runs is different: the
    # shape still yields a number, from fewer samples, and failing the whole
    # suite over one flaky request would be its own kind of wrong.
    empty = runner.results.empty_shapes()
    if empty:
        print(
            f"error: {len(empty)} shape(s) produced no measurement; "
            "results are incomplete.",
            file=sys.stderr,
        )
        return 1
    return 0

def main():
    """Entry point for the CLI command."""
    try:
        sys.exit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == "__main__":
    main()
