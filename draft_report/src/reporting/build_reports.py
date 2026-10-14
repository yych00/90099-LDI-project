"""Compile the authored draft sources without regenerating or overwriting their prose."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run numerical, authorship and PDF checks after compilation",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    # Derive tables from frozen pilot evidence, not whichever future run scores best.
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("prepare_tables.py"))], cwd=root, check=True
    )
    for engine, name in (("pdflatex", "report.tex"), ("xelatex", "report_cn.tex")):
        if not shutil.which(engine):
            parser.error(f"{engine} is missing; install a TeX distribution with ctex for Chinese.")
        for _ in range(2):
            subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", name], cwd=root, check=True
            )
    if args.check:
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("verify_draft.py"))], cwd=root, check=True
        )


if __name__ == "__main__":
    main()
