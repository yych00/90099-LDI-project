"""Backward-compatible report entry point; implementation lives in src/reporting."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "src" / "reporting" / "prepare_tables.py"), run_name="__main__")
