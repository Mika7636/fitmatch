"""One JSON object per line on stdout, so a request can be read after the fact.

The requirement for session 3 is a *structured* per-request log -- profile
summary, candidates after the hard filter, which soft constraints were
relaxed, the scoring mode, latency.  Those are fields, not prose, so they are
emitted as fields: a line is a JSON object and ``python -m json.tool`` or a
two-line pandas snippet can read a whole run of them.

Anything passed in ``logging``'s ``extra=`` ends up as a key on the object, so
the call sites stay ordinary logging calls.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

# Attributes LogRecord always has; anything else came from `extra=`.
_STANDARD = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {"asctime", "message", "taskName"}

LOGGER_NAME = "fitmatch.api"


class JsonLineFormatter(logging.Formatter):
    """Format a record as one JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach the JSON handler to the ``fitmatch`` logger tree, once.

    Idempotent: the TestClient in ``tests/test_api.py`` builds the app several
    times in one process, and duplicated handlers would duplicate every line.
    """
    logger = logging.getLogger("fitmatch")
    logger.setLevel(level)
    logger.propagate = False
    if not any(
        getattr(handler, "_fitmatch", False) for handler in logger.handlers
    ):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonLineFormatter())
        handler._fitmatch = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logging.getLogger(LOGGER_NAME)
