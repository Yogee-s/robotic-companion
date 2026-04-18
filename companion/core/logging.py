"""Coloured logging setup, usable with or without the coloredlogs package."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    level_num = getattr(logging, level.upper(), logging.INFO)
    try:
        import coloredlogs

        coloredlogs.install(
            level=level_num,
            fmt="%(asctime)s %(name)-24s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )
    except ImportError:
        logging.basicConfig(
            level=level_num,
            format="%(asctime)s %(name)-24s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )
