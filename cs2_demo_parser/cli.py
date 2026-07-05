from __future__ import annotations

import argparse
import logging

from .config import apply_cli_overrides, load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse CS2 .dem files into a MongoDB ML dataset.")
    parser.add_argument("--config", default="config.parse.json")
    parser.add_argument("--input-mode", choices=["faceit", "local"])
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--sample-every-ticks", type=int)
    parser.add_argument("--context-window-size", type=int)
    parser.add_argument("--context-window-count", type=int)
    parser.add_argument("--last-n", type=int)
    parser.add_argument("--date-from")
    parser.add_argument("--date-to")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Hide per-match inner progress bars; keep only the main demo progress bar.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = apply_cli_overrides(load_config(args.config), args)
    run_pipeline(config)
    return 0
