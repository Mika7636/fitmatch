"""``python -m src.api`` -- start the development server.

Exists so the project has one command that works from a plain checkout on
Windows without remembering uvicorn's import-string syntax.  ``run_api.bat``
calls this.
"""

from __future__ import annotations

import argparse

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.api",
        description="Serve the FitMatch API (FastAPI + uvicorn).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="restart on source changes (development only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("FitMatch API -> http://{}:{}/docs".format(args.host, args.port), flush=True)
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
