from __future__ import annotations

import argparse

from .literature import export_cache_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Utilities for BioML literature caches.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze",
        help="Export cached live-provider results to a frozen JSONL snapshot.",
    )
    freeze.add_argument(
        "--cache-dir",
        default=".cache/literature",
        help="Cache directory created by live literature providers.",
    )
    freeze.add_argument(
        "--out",
        required=True,
        help="Output JSONL snapshot path.",
    )
    freeze.add_argument(
        "--namespace",
        action="append",
        default=None,
        help="Only export a provider namespace. Can be passed more than once.",
    )

    args = parser.parse_args()
    if args.command == "freeze":
        count = export_cache_snapshot(
            cache_dir=args.cache_dir,
            output_path=args.out,
            namespaces=tuple(args.namespace) if args.namespace else None,
        )
        print(f"exported {count} cache records to {args.out}")


if __name__ == "__main__":
    main()
