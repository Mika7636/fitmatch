"""Block until GET /api/health reports ready, or give up.

Used by ``run_demo.bat`` between starting the API and starting the frontend.
A fixed ``timeout /t`` would have to be long enough for the worst case -- a
cold start that refits TF-IDF over 10,000 products -- which would make every
warm start, where the joblib cache loads in about 200 ms, feel broken.  Polling
the endpoint the API already exposes costs nothing and is honest about which
of the two happened.

Standard library only, and no dependency on ``src/``: it has to run before the
API is up and must not care whether the recommender imports cleanly.

    python scripts/wait_for_api.py --timeout 90
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000/api/health"


def poll(url: str, timeout: float, interval: float = 0.4) -> int:
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    last_status = ""

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            # Not listening yet, or listening but still importing. Both are
            # normal for the first second or two.
            time.sleep(interval)
            continue

        if body.get("ready"):
            cache = body.get("cache") or {}
            print(
                "  API ready in {:.1f}s  (catalogue {:,}, TF-IDF cache {})".format(
                    time.monotonic() - started,
                    int(body.get("catalogue_size") or 0),
                    "hit" if cache.get("hit_on_startup") else "rebuilt",
                )
            )
            return 0

        status = str(body.get("status") or "starting")
        if status != last_status:
            print("  ...{}".format(status))
            last_status = status
        time.sleep(interval)

    print(
        "  timed out after {:.0f}s waiting for {}".format(timeout, url),
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/wait_for_api.py",
        description="Poll the FitMatch health endpoint until it reports ready.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="health endpoint to poll")
    parser.add_argument(
        "--timeout", type=float, default=90.0, help="seconds before giving up"
    )
    args = parser.parse_args(argv)
    return poll(args.url, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
