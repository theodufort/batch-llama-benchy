"""Reporting: per-model CSV exports, graphs, and model recommendations."""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _parse_float(value: str) -> float | None:
    """Parse a float from a string, returning None on failure."""
    try:
        return float(value)
    except ValueError, TypeError:
        return None


def _load_rows(summary_csv: Path) -> list[dict]:
    """Read summary.csv and return a list of row dicts."""
    with open(summary_csv) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------
# CSV export
# ------------------------------------------------------------------


def export_per_model_csvs(results_dir: Path) -> list[Path]:
    """Split summary.csv into one CSV per model under results_dir/exports/."""
    rows = _load_rows(results_dir / "summary.csv")
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    export_dir = results_dir / "exports"
    export_dir.mkdir(exist_ok=True)
    paths = []

    fieldnames = [
        "model",
        "draft_n",
        "scenario",
        "pp",
        "tg",
        "depth",
        "tg_tok_per_s",
        "pp_tok_per_s",
        "result_file",
    ]

    for model in sorted(by_model):
        out = export_dir / ("%s_results.csv" % model)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(by_model[model])
        paths.append(out)

    return paths


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------


def _build_lookup(
    rows: list[dict],
) -> tuple[
    list[str],
    list[str],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[str, int],
]:
    """Return (models, scenarios, tg_avg, pp_avg, depth_map)."""
    models = sorted(set(r["model"] for r in rows))
    scenarios = sorted(set(r["scenario"] for r in rows))
    depth_map: dict[str, int] = {}

    tg_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    pp_acc: dict[tuple[str, str], list[float]] = defaultdict(list)

    for r in rows:
        tg = _parse_float(r["tg_tok_per_s"])
        pp = _parse_float(r["pp_tok_per_s"])
        depth = int(r.get("depth", 0))
        depth_map[r["scenario"]] = depth
        if tg is not None:
            tg_acc[(r["model"], r["scenario"])].append(tg)
        if pp is not None:
            pp_acc[(r["model"], r["scenario"])].append(pp)

    tg_avg = {k: statistics.mean(v) for k, v in tg_acc.items()}
    pp_avg = {k: statistics.mean(v) for k, v in pp_acc.items()}

    return models, scenarios, tg_avg, pp_avg, depth_map


# ------------------------------------------------------------------
# Graphs
# ------------------------------------------------------------------


def generate_graphs(results_dir: Path) -> list[Path]:
    """Generate visualization PNGs under results_dir/graphs/."""
    rows = _load_rows(results_dir / "summary.csv")
    models, scenarios, tg_avg, pp_avg, depth_map = _build_lookup(rows)

    graph_dir = results_dir / "graphs"
    graph_dir.mkdir(exist_ok=True)
    paths: list[Path] = []

    # 1. Grouped bar — TG throughput by scenario
    p = graph_dir / "tg_throughput_by_scenario.png"
    _plot_grouped_bar(scenarios, models, tg_avg, "TG Throughput (tok/s)", p)
    paths.append(p)

    # 2. Grouped bar — PP throughput by scenario
    p = graph_dir / "pp_throughput_by_scenario.png"
    _plot_grouped_bar(scenarios, models, pp_avg, "PP Throughput (tok/s)", p)
    paths.append(p)

    # 3. Line — TG throughput vs context depth
    ctx_scenarios = [s for s in scenarios if "ctx-" in s]
    if ctx_scenarios:
        p = graph_dir / "tg_throughput_vs_depth.png"
        _plot_depth_line(ctx_scenarios, models, tg_avg, depth_map, p)
        paths.append(p)

    # 4. Radar — relative speed comparison
    p = graph_dir / "speed_radar.png"
    _plot_radar(scenarios, models, tg_avg, p)
    paths.append(p)

    # 5. Heatmap — TG throughput
    p = graph_dir / "tg_heatmap.png"
    _plot_heatmap(scenarios, models, tg_avg, p)
    paths.append(p)

    return paths


def _plot_grouped_bar(
    scenarios: list[str],
    models: list[str],
    data: dict[tuple[str, str], float],
    ylabel: str,
    out: Path,
) -> None:
    x = list(range(len(scenarios)))
    width = 0.7 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 0.65), 6))
    for i, model in enumerate(models):
        vals = [data.get((model, s), 0) for s in scenarios]
        ax.bar([p + i * width for p in x], vals, width, label=model, edgecolor="white")

    offset = width * (len(models) - 1) / 2
    ax.set_xticks([p + offset for p in x])
    ax.set_xticklabels(
        [s.replace("-", "\n") for s in scenarios], rotation=0, fontsize=7
    )
    ax.set_ylabel(ylabel)
    ax.set_title("%s by Scenario" % ylabel)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.StrFormat("%.0f"))
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)


def _plot_depth_line(
    scenarios: list[str],
    models: list[str],
    data: dict[tuple[str, str], float],
    depth_map: dict[str, int],
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        depths = [depth_map.get(s, 0) / 1024.0 for s in scenarios]
        vals = [data.get((model, s), 0) for s in scenarios]
        ax.plot(depths, vals, "-o", label=model, markersize=5)

    ax.set_xlabel("Context Depth (K tokens)")
    ax.set_ylabel("TG Throughput (tok/s)")
    ax.set_title("TG Throughput vs Context Depth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)


def _plot_radar(
    scenarios: list[str],
    models: list[str],
    data: dict[tuple[str, str], float],
    out: Path,
) -> None:
    """Normalized radar chart showing relative speed across scenarios."""
    max_vals: dict[str, float] = {}
    for s in scenarios:
        max_vals[s] = max((data.get((m, s), 0) for m in models), default=1)

    N = len(scenarios)
    angles = [2 * math.pi * n / N for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    for model in models:
        vals = [data.get((model, s), 0) / max_vals[s] for s in scenarios]
        vals += vals[:1]
        ax.plot(angles, vals, "-o", label=model, linewidth=1.5)
        ax.fill(angles, vals, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([s.replace("-", "\n") for s in scenarios], fontsize=7)
    ax.set_title("Relative Speed Comparison (normalized)", va="top", y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=7)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(
    scenarios: list[str],
    models: list[str],
    data: dict[tuple[str, str], float],
    out: Path,
) -> None:
    """Heatmap: scenarios x models, colored by TG throughput."""
    matrix = [[data.get((m, s), 0) for m in models] for s in scenarios]

    fig, ax = plt.subplots(
        figsize=(max(8, len(models) * 1.5), max(6, len(scenarios) * 0.4))
    )
    cax = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(list(range(len(models))))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(list(range(len(scenarios))))
    ax.set_yticklabels([s.replace("-", "\n") for s in scenarios], fontsize=7)
    ax.set_title("TG Throughput Heatmap (tok/s)")

    for i in range(len(scenarios)):
        for j in range(len(models)):
            ax.text(j, i, "%.0f" % matrix[i][j], ha="center", va="center", fontsize=6)

    fig.colorbar(cax, label="tok/s")
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# Recommendations
# ------------------------------------------------------------------


def recommend(results_dir: Path) -> str:
    """Analyze results and return a plain-text recommendation report."""
    rows = _load_rows(results_dir / "summary.csv")
    models, scenarios, tg_avg, pp_avg, depth_map = _build_lookup(rows)

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  MODEL RECOMMENDATIONS")
    lines.append("=" * 60)

    # 1. Fastest generation overall
    def avg_tg(m: str) -> float:
        vals = [tg_avg.get((m, s), 0) for s in scenarios]
        return statistics.mean(vals) if vals else 0

    winner = max(models, key=avg_tg)
    lines.append("")
    lines.append(
        "[1] FASTEST OVERALL (TG tok/s): %s  (%.1f avg tok/s)"
        % (winner, avg_tg(winner))
    )

    # 2. Best for short prompts (baseline scenarios)
    baseline = [s for s in scenarios if s.startswith("baseline")]
    if baseline:
        best_bl = max(
            models,
            key=lambda m: statistics.mean([tg_avg.get((m, s), 0) for s in baseline]),
        )
        lines.append("[2] BEST FOR SHORT PROMPTS (baseline): %s" % best_bl)

    # 3. Best under context pressure
    ctx = [s for s in scenarios if s.startswith("ctx-")]
    if ctx:
        shallow = [s for s in ctx if depth_map.get(s, 0) <= 16384]
        medium = [s for s in ctx if 16384 < depth_map.get(s, 0) <= 65536]
        deep = [s for s in ctx if depth_map.get(s, 0) > 65536]

        def best_for(sc: list[str]) -> str:
            if not sc:
                return "N/A"
            return max(
                models,
                key=lambda m: statistics.mean([tg_avg.get((m, s), 0) for s in sc]),
            )

        lines.append("[3] BEST UNDER CONTEXT PRESSURE:")
        if shallow:
            lines.append("     Shallow context (<=16K):  %s" % best_for(shallow))
        if medium:
            lines.append("     Medium context (16-64K): %s" % best_for(medium))
        if deep:
            lines.append("     Deep context (>64K):     %s" % best_for(deep))

    # 4. Best prefill speed
    pp_heavy = [s for s in scenarios if "pp-heavy" in s]
    if pp_heavy:
        best_pp = max(
            models,
            key=lambda m: statistics.mean([pp_avg.get((m, s), 0) for s in pp_heavy]),
        )
        lines.append("[4] BEST PREFILL SPEED (pp-heavy): %s" % best_pp)

    # 5. Best long generation under context
    longgen = [s for s in scenarios if "longgen" in s]
    if longgen:
        best_lg = max(
            models,
            key=lambda m: statistics.mean([tg_avg.get((m, s), 0) for s in longgen]),
        )
        lines.append("[5] BEST LONG GENERATION UNDER CONTEXT: %s" % best_lg)

    # 6. Most consistent (lowest std dev)
    def consistency(m: str) -> float:
        vals = [tg_avg.get((m, s), 0) for s in scenarios]
        return statistics.stdev(vals) if len(vals) > 1 else 0

    most_consistent = min(models, key=consistency)
    lines.append(
        "[6] MOST CONSISTENT: %s  (std dev: %.1f)"
        % (most_consistent, consistency(most_consistent))
    )

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------


def generate_report(results_dir: Path) -> None:
    """Generate all exports, graphs, and print recommendations."""
    print("[bench] Generating reports...")

    csvs = export_per_model_csvs(results_dir)
    print("[ok] Per-model CSVs exported: %d" % len(csvs))

    graphs = generate_graphs(results_dir)
    print("[ok] Graphs generated: %d" % len(graphs))

    print()
    print(recommend(results_dir))
