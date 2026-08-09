# batch-bench-llama

Benchmark speculative decoding draft configurations using [llama-benchy](https://github.com/.../llama-benchy). Runs a matrix of models against a suite of predefined scenarios (baseline, context depth, heavy prefill, long generation) and produces a summary CSV with throughput metrics.

## Requirements

- **Python &ge;3.14**
- **uv** (for `uvx` and dependency management)
- A running LLM API server (e.g., llama.cpp) with the models you want to benchmark loaded on demand

## Installation

```bash
git clone <repo-url>
cd batch-llama-benchy
pip install -e .
```

## Usage

```bash
batch-bench-llama --models MODEL1,MODEL2 [--host HOST] [--port PORT] [--runs N]
```

### Arguments

| Flag      | Default           | Description                                      |
|-----------|-------------------|--------------------------------------------------|
| `--models`| *(required)*      | Comma-separated model names (at least 2)         |
| `--host`  | `127.0.0.1`     | LLM server hostname                              |
| `--port`  | `8080`          | LLM server port                                  |
| `--runs`  | `3`               | Number of benchmark runs per scenario            |

### Example

```bash
batch-bench-llama --models llama-3.1-8b-d3,llama-3.1-8b-d5,llama-3.1-8b-d7 --host 10.0.0.5
```

This will:

1. Iterate over each model, waiting for the server to load it.
2. Run every scenario (see below) against each model.
3. Output a formatted summary table to the console.
4. Save all results to a timestamped directory: `bench_results_YYYYMMDD_HHMMSS/`.

## Scenarios

| Category | Description |
|---|---|
| **baseline** | Short, medium, and long generation with no context depth |
| **ctx-\*k** | Increasing context depth (4k&ndash;120k) with fixed prompt/generation lengths |
| **pp-heavy-\*k** | Heavy prefill (4096 tokens) at various context depths |
| **longgen-ctx-\*k** | Long generation (1024 tokens) under context pressure |

Customize scenarios by editing `SCENARIOS` in `src/bench_llama/config.py`.

## Output

Each run creates a directory like `bench_results_20260809_143000/` containing:

- **summary.csv** &mdash; all results in CSV format (model, draft\_n, scenario, pp, tg, depth, tg\_tok/s, pp\_tok/s, result\_file)
- **`<model>/`** &mdash; per-model subdirectory with individual `.json` result files and `.log` files per scenario

The CLI also prints a formatted table showing:
- TG and PP tokens/second per model &times; scenario
- Best draft configuration per scenario (by TG tok/s)

## Dependencies

- [llama-benchy](https://github.com/.../llama-benchy) &mdash; invoked via `uvx` at runtime (no install needed)
- `requests`, `tabulate` &mdash; installed with the package
