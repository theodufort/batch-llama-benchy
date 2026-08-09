"""Core benchmark runner: invokes llama-benchy for each model × scenario."""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from .config import Scenario, extract_draft_n
from .server import wait_for_server


def run_benchmarks(
    models: list[str],
    scenarios: list[Scenario],
    base_url: str,
    runs: int = 3,
) -> Path:
    """Run all benchmarks and return the results directory path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("./bench_results_%s" % timestamp)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = results_dir / "summary.csv"
    summary_csv.write_text(
        "model,draft_n,scenario,pp,tg,depth,tg_tok_per_s,pp_tok_per_s,result_file\n"
    )

    total = len(models) * len(scenarios)
    done = 0

    for model in models:
        draft_n = extract_draft_n(model)
        print("[bench] ========================================================")
        print("[bench]  Model: %s  (spec-draft-n-max=%s)" % (model, draft_n))
        print("[bench] ========================================================")

        # Wait for server to be ready with this model
        wait_for_server(model, base_url)

        model_dir = results_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        for scenario in scenarios:
            done += 1
            pct = done * 100 // total
            print(
                "[%d/%d %d%%] %s :: %s "
                "(pp=%d tg=%d depth=%d)"
                % (
                    done,
                    total,
                    pct,
                    model,
                    scenario.label,
                    scenario.pp,
                    scenario.tg,
                    scenario.depth,
                )
            )

            outfile = model_dir / f"{scenario.label}.json"

            # Build llama-benchy command
            cmd = [
                "uvx",
                "llama-benchy",
                "--base-url",
                base_url,
                "--model",
                model,
                "--pp",
                str(scenario.pp),
                "--tg",
                str(scenario.tg),
                "--depth",
                str(scenario.depth),
                "--runs",
                str(runs),
                "--no-cache",
                "--skip-coherence",
                "--format",
                "json",
                "--save-result",
                str(outfile),
            ]

            try:
                with open("%s.log" % outfile, "w") as log_file:
                    subprocess.run(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                print("[ok]    → saved: %s" % outfile)

                # Extract metrics
                tg_tok, pp_tok = extract_metrics(outfile)
                line = "%s,%s,%s,%d,%d,%d,%s,%s,%s\n" % (
                    model,
                    draft_n,
                    scenario.label,
                    scenario.pp,
                    scenario.tg,
                    scenario.depth,
                    tg_tok,
                    pp_tok,
                    outfile,
                )
                summary_csv.write_text(summary_csv.read_text() + line)

            except subprocess.CalledProcessError:
                print(
                    "[warn]  ✗ bench failed for %s::%s — check %s.log"
                    % (model, scenario.label, outfile)
                )
                line = "%s,%s,%s,%d,%d,%d,FAILED,FAILED,%s.log\n" % (
                    model,
                    draft_n,
                    scenario.label,
                    scenario.pp,
                    scenario.tg,
                    scenario.depth,
                    outfile,
                )
                summary_csv.write_text(summary_csv.read_text() + line)

            time.sleep(2)  # Let KV/VRAM settle

    return results_dir


def extract_metrics(outfile: Path) -> tuple:
    """Parse llama-benchy JSON output and return (tg_tok_per_s, pp_tok_per_s)."""
    try:
        with open(outfile) as f:
            d = json.load(f)
        # llama-benchy 0.4+ format: benchmarks[0].tg_throughput.mean
        benchmarks = d.get("benchmarks", [])
        if benchmarks:
            b = benchmarks[0]
            tg = b.get("tg_throughput", {}).get("mean", "N/A")
            pp = b.get("pp_throughput", {}).get("mean", "N/A")
            return (str(tg), str(pp))
        # Fallback: older format with results[] or top-level keys
        if isinstance(d, list):
            r = d[0]
        elif "results" in d:
            r = d["results"][0]
        else:
            r = d
        tg = r.get("tg_tok_per_s", r.get("tg", {}).get("avg", "N/A"))
        pp = r.get("pp_tok_per_s", r.get("pp", {}).get("avg", "N/A"))
        return (str(tg), str(pp))
    except Exception:
        return ("N/A", "N/A")
