"""echo_cli: Standardized command-line interface for ECHO 4D."""

from echo_cli.main import main
from echo_cli.parser import build_parser

__all__ = ["main", "build_parser"]
