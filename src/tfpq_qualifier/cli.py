"""Command-line entrypoint."""

from __future__ import annotations

import argparse

from .artifacts import generate_artifacts
from .pipeline import run


def main(argv: list[str] | None = None) -> int:
    """Run the guided non-coder pipeline."""
    parser = argparse.ArgumentParser(
        description="Qualify time-frequency PQ descriptors on controlled synthetic signals."
    )
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument(
        "--reproduction", action="store_true", help="Generate all manuscript artifacts"
    )
    parser.add_argument("--non-interactive", action="store_true", help="Run without prompts")
    args = parser.parse_args(argv)
    result = run(output_dir=args.output)
    if args.reproduction:
        generate_artifacts(result)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
