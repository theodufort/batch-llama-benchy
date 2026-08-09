"""CLI entry point for batch-bench-llama."""

import argparse
from pathlib import Path

from .config import SCENARIOS, extract_draft_n, validate_models
from .reporter import generate_report
from .runner import extract_metrics, run_benchmarks
from .server import validate_models_on_server

# Build a lookup from scenario label to Scenario
_SCENARIO_MAP = {s.label: s for s in SCENARIOS}


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
        help="Comma-separated list of model names (at least 2 required)",
    )
    parser.add_argument(
        "--results-dir",
        help="Path to existing results directory (use with --skip-bench to regenerate reports only)",
    )
    parser.add_argument(
        "--skip-bench",
        action="store_true",
        help="Skip benchmark inference; rebuild summary from JSON files and regenerate reports",
    )

    args = parser.parse_args()

    # Skip-bench mode: rebuild summary from existing JSON results
    if args.skip_bench:
        if not args.results_dir:
            raise SystemExit("[fail] --results-dir is required when using --skip-bench")
        results_dir = Path(args.results_dir)
        summary_csv = results_dir / "summary.csv"

        # Rebuild summary.csv from JSON result files
        print(f"[bench] Rebuilding summary from JSON results in: {results_dir}")
        header = (
            "model,draft_n,scenario,pp,tg,depth,tg_tok_per_s,pp_tok_per_s,result_file\n"
        )
        summary_csv.write_text(header)

        model_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
        rebuilt = 0
        for model_dir in model_dirs:
            model = model_dir.name
            draft_n = extract_draft_n(model)
            for json_file in sorted(model_dir.glob("*.json")):
                scenario_label = json_file.stem
                tg, pp = extract_metrics(json_file)
                sc = _SCENARIO_MAP.get(scenario_label)
                pp_val = str(sc.pp) if sc else ""
                tg_val = str(sc.tg) if sc else ""
                depth_val = str(sc.depth) if sc else ""
                line = f"{model},{draft_n},{scenario_label},{pp_val},{tg_val},{depth_val},{tg},{pp},{json_file.name}\n"
                summary_csv.write_text(summary_csv.read_text() + line)
                rebuilt += 1

        print(f"[ok] Rebuilt {rebuilt} rows in summary.csv")
        print()
        print_summary(summary_csv)
        print()
        generate_report(results_dir)
        print()
        print(f"[ok] Reports regenerated in: {results_dir}/")
        return

    # Normal mode: run benchmarks
    if not args.models:
        raise SystemExit("[fail] --models is required (or use --skip-bench)")

    # Parse models
    models = [m.strip() for m in args.models.split(",")]
    validate_models(models)

    base_url = f"http://{args.host}:{args.port}/v1"

    # Validate models exist on server before starting
    validate_models_on_server(models, base_url)

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

    # Generate reports: per-model CSVs, graphs, recommendations
    print()
    generate_report(results_dir)

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
