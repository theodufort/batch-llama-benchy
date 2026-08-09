"""CLI entry point for bench-llama."""

import argparse
from pathlib import Path

from .config import SCENARIOS, validate_models
from .runner import run_benchmarks


def main():
    parser = argparse.ArgumentParser(
        prog="batch-bench-llama",
        description="Benchmark speculative decoding drafts with llama-benchy",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per scenario",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated list of model names (at least 2 required)",
    )

    args = parser.parse_args()

    # Parse models
    models = [m.strip() for m in args.models.split(",")]
    validate_models(models)

    base_url = f"http://{args.host}:{args.port}/v1"

    print(f"[bench] Models:          {models}")
    print(f"[bench] Base URL:        {base_url}")
    print(f"[bench] Runs per test:   {args.runs}")
    print()

    # Run benchmarks
    results_dir = Path(
        run_benchmarks(
            models=models,
            scenarios=SCENARIOS,
            base_url=base_url,
            runs=args.runs,
        )
    )

    # Print summary
    summary_csv = results_dir / "summary.csv"
    print_summary(summary_csv)

    print()
    print(f"[ok] Full results in: {results_dir}/")


def print_summary(summary_csv: Path):
    """Print a formatted summary table and best-per-model analysis."""
    import csv

    from tabulate import tabulate

    print("\n[bench] ========================================================")
    print("[bench] BENCHMARK COMPLETE")
    print("[bench] ========================================================")
    print(f"[bench] Summary CSV: {summary_csv}")
    print()

    # Print table
    rows = []
    with open(summary_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                [
                    row["model"],
                    row["scenario"],
                    row.get("tg_tok_per_s", "N/A"),
                    row.get("pp_tok_per_s", "N/A"),
                ]
            )
    print("--- Summary (tg_tok_per_s) ---")
    print(
        tabulate(
            rows,
            headers=["Model", "Scenario", "TG tok/s", "PP tok/s"],
            tablefmt="simple",
        )
    )

    # Best draft per scenario
    print()
    print("--- Best draft config per scenario (by TG tok/s) ---")
    best = {}  # scenario -> (tg_tok/s, model, draft_n)
    with open(summary_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row["scenario"]
            try:
                tg = float(row["tg_tok_per_s"])
            except ValueError, KeyError:
                continue
            if scenario not in best or tg > best[scenario][0]:
                best[scenario] = (tg, row["model"], row["draft_n"])

    lines = []
    for sc, (tg, model, dn) in sorted(best.items()):
        lines.append([sc, model, dn, f"{tg:.1f}"])
    print(
        tabulate(
            lines,
            headers=["Scenario", "Best Model", "draft_n", "TG tok/s"],
            tablefmt="simple",
        )
    )


if __name__ == "__main__":
    main()
