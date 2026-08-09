# Contributing

## Local setup

```bash
git clone <repo-url>
cd batch-llama-benchy
pip install -e .
```

This installs the package in editable mode. Use `uv run batch-bench-llama` to invoke the CLI.

## Testing

Run a quick benchmark against a local llama.cpp server:

```bash
uv run batch-bench-llama --models llama-3.1-8b-d3,llama-3.1-8b-d5 --runs 1
```

Results (CSVs, graphs, and recommendations) will be written to `./bench_results_<timestamp>/`.

There are no unit tests yet — contributions adding tests are welcome.

## Code style

- **Python &ge;3.14** only
- **Line length**: 88 characters (Ruff default)
- **String formatting**: Use `%-style` formatting (`"%s" % val`), not f-strings. Ruff rule `UP031` is disabled in `ruff.toml`.
- **Type hints**: Always annotate function signatures and return types.
- **Docstrings**: Module-level one-liner + docstring on every public function.
- **Imports**: Standard library &rarr; third-party &rarr; local (relative imports with `.`).
- **Naming**: `snake_case` for functions and variables. Descriptive names over abbreviations.
- **Console output**: Prefix log lines with `[bench]`, `[ok]`, `[warn]`, or `[fail]` for consistency.
- **Error handling**: Prefer `raise SystemExit("...")` for fatal CLI errors over `sys.exit()`.

### Ruff

We use Ruff for linting and formatting. Run before committing:

```bash
uvx ruff check --fix
```

Active rules (in `ruff.toml`): `E`, `F`, `W`, `I`, `UP`, `B`, `SIM`.
Ignored: `E501` (line-length handled by ruff format), `UP031` (we prefer `%` formatting).

## Project structure

```
src/bench_llama/
├── __init__.py      # package marker
├── cli.py           # CLI entry point (argparse)
├── config.py        # scenarios, validation helpers
├── runner.py        # benchmark orchestration (invokes llama-benchy)
├── server.py        # health-check / wait-for-model helpers
└── reporter.py      # CSV export, graphs, recommendations
```

## Adding scenarios

Edit `SCENARIOS` in `src/bench_llama/config.py`. Each `Scenario` is a `NamedTuple`:

```python
Scenario("label", pp_tokens, tg_tokens, context_depth)
```

Group related scenarios with `# ---- category ----` comment headers.

## Pull requests

1. Keep changes focused — one feature or fix per PR.
2. Run `uvx ruff check --fix` before pushing.
3. Update `README.md` if CLI flags or output format changes.
