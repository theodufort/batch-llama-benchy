"""Configuration: scenarios and validation helpers."""

from typing import NamedTuple


class Scenario(NamedTuple):
    label: str
    pp: int  # prompt processing tokens
    tg: int  # tokens to generate
    depth: int  # context depth tokens


SCENARIOS: list[Scenario] = [
    # ---- context filling — same short prompt/gen, increasing depth ----
    Scenario("ctx-16k", 128, 256, 16384),
    Scenario("ctx-32k", 128, 256, 32768),
    Scenario("ctx-64k", 128, 256, 65536),
    Scenario("ctx-96k", 128, 256, 98304),
    Scenario("ctx-120k", 128, 256, 122880),
    Scenario("ctx-128k", 128, 256, 131072),
    # ---- long generation under context pressure ----
    Scenario("longgen-ctx-16k", 256, 1024, 16384),
    Scenario("longgen-ctx-64k", 256, 1024, 65536),
    Scenario("longgen-ctx-128k", 256, 1024, 131072),
    # ---- realistic agentic coding: large prefill + long gen at deep context ----
    Scenario("agentic-64k", 4096, 1024, 65536),
    Scenario("agentic-96k", 4096, 1024, 98304),
    Scenario("agentic-128k", 4096, 1024, 131072),
]


def validate_models(models: list[str]) -> None:
    """Raise if no models are provided."""
    if len(models) < 1:
        raise SystemExit(
            "[fail] At least 1 model is required. Provide them with --models MODEL1,MODEL2,..."
        )


def extract_draft_n(model_name: str) -> str:
    """Extract the draft number suffix (e.g. 'd5' -> '5')."""
    return model_name.split("-d")[-1]
