"""Server health-check helpers."""

import time

import requests


def wait_for_server(
    model: str,
    base_url: str,
    max_wait: int = 120,
) -> None:
    """Block until the model name appears in /v1/models."""
    print(f"[bench] Waiting for model '{model}' at {base_url} ...")
    elapsed = 0
    while elapsed < max_wait:
        try:
            resp = requests.get(f"{base_url}/models", timeout=5)
            if resp.ok and model in resp.text:
                print(f"[ok]    Model '{model}' is live.")
                return
        except requests.RequestException:
            pass
        time.sleep(3)
        elapsed += 3
    raise SystemExit(f"[fail] Timed out waiting for model '{model}' after {max_wait}s")
