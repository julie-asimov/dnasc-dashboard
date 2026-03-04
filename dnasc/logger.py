"""
dnasc/logger.py
────────────────
Centralised logging setup.
Every module gets a named logger via:

    from dnasc.logger import get_logger
    log = get_logger(__name__)

Script Server captures stdout, so we emit to both a rotating file
and stdout so logs appear in the Script Server run output.
"""

from __future__ import annotations
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Log file lives next to the scripts/ directory ────────────────────────────
_LOG_DIR  = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "dnasc.log"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Format ────────────────────────────────────────────────────────────────────
_FMT  = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# ── Root logger — configured once ─────────────────────────────────────────────
def _configure_root() -> None:
    root = logging.getLogger("dnasc")
    if root.handlers:          # already configured (e.g. in tests)
        return

    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    # Rotating file — 10 MB × 5 backups
    fh = RotatingFileHandler(_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Stdout — INFO and above (visible in Script Server run output)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)


_configure_root()


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'dnasc' namespace.

    Usage:
        log = get_logger(__name__)
        log.info("Fetched %d rows", len(df))
    """
    # Strip absolute module path so names stay tidy, e.g. 'dnasc.extractors.bios'
    if not name.startswith("dnasc"):
        name = f"dnasc.{name}"
    return logging.getLogger(name)
