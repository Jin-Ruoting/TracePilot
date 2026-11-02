"""Single CLI entrypoint for the code-research workflow."""
from __future__ import annotations

import argparse

from .research.cli import add_parser, handle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tracepilot", description="Evidence-backed code research and reproducible experiments.")
    add_parser(parser.add_subparsers(dest="action", required=True))
    return parser


def main(argv: list[str] | None = None) -> int:
    return handle(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
