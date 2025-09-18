"""Convenience launcher for the CHMD trainer."""
from __future__ import annotations

import sys

from chmd_trainer import main


def run() -> None:
    """Entry point used by IDE run configurations."""
    try:
        main()
    except KeyboardInterrupt:
        # Allow a clean exit when stopping from an IDE or terminal.
        sys.exit(130)


if __name__ == "__main__":
    run()
