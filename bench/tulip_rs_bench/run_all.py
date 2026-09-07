"""
Entry point for tulip-rs-bench.

Installed as the `tulip-rs-bench` console script via pyproject.toml.

Usage:
    tulip-rs-bench                   # all indicators, stdout only
    tulip-rs-bench ema rsi macd      # specific indicators
    tulip-rs-bench psar+             # psar and everything after it, alphabetically
    tulip-rs-bench psar+ ema         # range start plus an extra specific indicator
    BENCHMARK_LOG_TO_DB=1 tulip-rs-bench   # write to indicator_benchmark DB
    python -m tulip_rs_bench.run_all       # equivalent alternative
"""

from __future__ import annotations

import importlib
import os
import sys
import time

from tulip_rs_bench.common import (
    LOG_TO_DB,
    BenchmarkDef,
    BenchmarkLogger,
    load_stock_data,
    run_benchmark,
)

_INDICATORS_PKG = "tulip_rs_bench.indicators"
_INDICATORS_DIR = os.path.join(os.path.dirname(__file__), "indicators")


def _discover() -> list[tuple[str, BenchmarkDef]]:
    """Return (module_stem, BENCHMARK) for every bench_*.py in indicators/."""
    found = []
    for fname in sorted(os.listdir(_INDICATORS_DIR)):
        if not (fname.startswith("bench_") and fname.endswith(".py")):
            continue
        mod_name = f"{_INDICATORS_PKG}.{fname[:-3]}"
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "BENCHMARK"):
            print(
                f"[warn] {fname} has no BENCHMARK variable — skipping", file=sys.stderr
            )
            continue
        found.append((fname[:-3], mod.BENCHMARK))
    return found


def _filter_benchmarks(
    all_benchmarks: list[tuple[str, BenchmarkDef]], filter_names: list[str]
) -> list[tuple[str, BenchmarkDef]]:
    """
    Select benchmarks from filter_names.

    Plain names (e.g. "ema") match a single indicator by BenchmarkDef.name.
    A name suffixed with "+" (e.g. "psar+") is a range start: it selects
    that indicator and every indicator after it in alphabetical order
    (by BenchmarkDef.name), letting you resume a long run from where it
    left off. Range starts and plain names can be combined freely; results
    are de-duplicated and preserve discovery order for plain names and
    alphabetical order for range-selected indicators.
    """
    range_starts = [n[:-1] for n in filter_names if n.endswith("+")]
    exact_names = [n for n in filter_names if not n.endswith("+")]

    selected: list[tuple[str, BenchmarkDef]] = []
    seen: set[str] = set()

    if range_starts:
        by_name = sorted(all_benchmarks, key=lambda nb: nb[1].name.lower())
        for start in range_starts:
            start_lower = start.lower()
            idx = next(
                (i for i, (_n, b) in enumerate(by_name) if b.name.lower() == start_lower),
                None,
            )
            if idx is None:
                print(
                    f"[error] No benchmark named '{start}' found for range start '{start}+'",
                    file=sys.stderr,
                )
                sys.exit(1)
            for n, b in by_name[idx:]:
                if b.name not in seen:
                    seen.add(b.name)
                    selected.append((n, b))

    if exact_names:
        for n, b in all_benchmarks:
            if b.name in exact_names and b.name not in seen:
                seen.add(b.name)
                selected.append((n, b))

    return selected


def main(filter_names: list[str] | None = None) -> None:
    # When invoked as a console script the entry point calls main() with no
    # arguments, so fall back to reading sys.argv directly.
    if filter_names is None and len(sys.argv) > 1:
        filter_names = sys.argv[1:]
    print("=" * 64)
    print("  tulip-rs Python Benchmark Suite")
    print("=" * 64)

    print("\n[1/3] Loading stock data …")
    stocks = load_stock_data()
    if not stocks:
        print("[error] No stock data — is the stocks DB running?", file=sys.stderr)
        sys.exit(1)

    logger: BenchmarkLogger | None = None
    if LOG_TO_DB:
        print("\n[2/3] Connecting to benchmark DB …")
        logger = BenchmarkLogger()
        logger.start_run()
    else:
        print("\n[2/3] DB logging disabled (BENCHMARK_LOG_TO_DB=0) — stdout only")

    print("\n[3/3] Running benchmarks …")
    all_benchmarks = _discover()
    if filter_names:
        all_benchmarks = _filter_benchmarks(all_benchmarks, filter_names)
        if not all_benchmarks:
            print(f"[error] No benchmarks matched: {filter_names}", file=sys.stderr)
            sys.exit(1)

    t_start = time.perf_counter()
    for _stem, defn in all_benchmarks:
        run_benchmark(defn, stocks, logger)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 64}")
    print(f"  Finished {len(all_benchmarks)} indicator(s) in {elapsed:.1f}s")
    if logger:
        print(f"  Results written to DB (run_id={logger.run_id})")
        logger.close()
    print("=" * 64)


if __name__ == "__main__":
    main(filter_names=sys.argv[1:] or None)
