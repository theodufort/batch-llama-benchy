"""Configuration: scenarios and validation helpers."""

from typing import NamedTuple


class Scenario(NamedTuple):
    label: str
    pp: int  # prompt processing tokens
    tg: int  # tokens to generate
    depth: int  # context depth tokens


SCENARIOS: list[Scenario] = [
    # ---- baseline (no context) ----
    Scenario("baseline-short", 128, 64, 0),
    Scenario("baseline-med", 512, 256, 0),
    Scenario("baseline-long-gen", 128, 512, 0),
    Scenario("baseline-long-gen-xl", 128, 1024, 0),
    # ---- context filling — same short prompt/gen, increasing depth ----
    Scenario("ctx-4k", 128, 256, 4096),
    Scenario("ctx-8k", 128, 256, 8192),
    Scenario("ctx-16k", 128, 256, 16384),
    Scenario("ctx-32k", 128, 256, 32768),
    Scenario("ctx-64k", 128, 256, 65536),
    Scenario("ctx-96k", 128, 256, 98304),
    Scenario("ctx-120k", 128, 256, 122880),
    # ---- heavy pp (prefill stress) at various depths ----
    Scenario("pp-heavy-0k", 4096, 256, 0),
    Scenario("pp-heavy-16k", 4096, 256, 16384),
    Scenario("pp-heavy-64k", 4096, 256, 65536),
    # ---- long generation under context pressure ----
    Scenario("longgen-ctx-16k", 256, 1024, 16384),
    Scenario("longgen-ctx-64k", 256, 1024, 65536),
]


def validate_models(models: list[str]) -> None:
    """Raise if fewer than 2 models are provided."""
    if len(models) < 2:
        raise SystemExit(
            "[fail] At least 2 models are required. Provide them with --models MODEL1,MODEL2,..."
        )


def extract_draft_n(model_name: str) -> str:
    """Extract the draft number suffix (e.g. 'd5' -> '5')."""
    return model_name.split("-d")[-1]
