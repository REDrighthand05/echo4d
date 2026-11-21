"""CLI main entry point for ECHO."""

import sys
from typing import List, Optional

from echo_cli.parser import build_parser
from echo_cli.runner import (
    handle_depth,
    handle_export,
    handle_run,
    handle_serve,
    handle_status,
)


def main(argv: Optional[List[str]] = None) -> int:
    """Parse command line arguments and dispatch to the corresponding runner."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    dispatch = {
        "run": handle_run,
        "depth": handle_depth,
        "serve": handle_serve,
        "export": handle_export,
        "status": handle_status,
    }

    handler = dispatch.get(args.subcommand)
    if not handler:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
