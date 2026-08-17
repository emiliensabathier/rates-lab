"""Command-line entry point: ``python -m rlab``.

This always performs a live fetch (subject to the on-disk cache in ``--cache-dir``), so its
default output deliberately does *not* point at the committed ``reports/rates.html``: that
file is the artefact the regression suite verifies against the frozen fixture
(``tests/fixtures/``), and a live run silently overwriting it would break that guarantee for
anyone who ran the documented command. Reproduce the committed page with
``scripts/build_frozen_report.py`` instead; pass ``--output reports/rates.html`` here only if
you deliberately want to replace it with fresh, unfrozen numbers.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from rlab.pipeline import run
from rlab.report.build import render

DEFAULT_OUTPUT = "reports/rates.local.html"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the rates report from live data")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--refresh", action="store_true", help="ignore the cached data")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    result = run(cache_dir=Path(args.cache_dir), refresh=args.refresh)
    html = render(result, generated_on=datetime.now(UTC).date().isoformat())

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    print(f"wrote {destination} ({len(html):,} bytes)")

    if result.refusals:
        for date, reason in sorted(result.refusals.items()):
            print(f"  refused {date}: {reason}")
        # A refusal is not a crash -- the page still renders from the months that did build
        # -- but it is not success either. Exiting 0 would be a silent failure to any script
        # or CI step that only checks the exit code.
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
