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


def validate_models_on_server(models: list[str], base_url: str) -> None:
    """Check that all requested models are available via /v1/models."""
    try:
        resp = requests.get("%s/models" % base_url, timeout=5)
        if not resp.ok:
            raise SystemExit(
                "[fail] Server returned HTTP %d — is it running?" % resp.status_code
            )
        data = resp.json()
        available = {m.get("id") for m in data.get("data", [])}
        missing = [m for m in models if m not in available]
        if missing:
            raise SystemExit(
                "[fail] Model(s) not found on server: %s\n"
                "Available: %s" % (", ".join(missing), ", ".join(sorted(available)))
            )
        print("[ok]    All %d models verified on server." % len(models))
    except requests.RequestException as e:
        raise SystemExit("[fail] Cannot reach server at %s: %s" % (base_url, e)) from e
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit("[fail] Unexpected error checking models: %s" % e) from e
