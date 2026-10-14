"""Package source, small reproducibility evidence and report sources for separate code upload."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve() if args.output else root / "code.zip"
    evidence = root / "src" / "evidence"
    if output == evidence or evidence in output.parents:
        parser.error("The code archive cannot be written inside frozen evidence.")
    entries = []
    for path in (root / "src").rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == output:
            continue
        relative = path.relative_to(root)
        # Generated checkpoints and caches are not implementation files. Excluding
        # them keeps the grader's code upload small while data can be reconstructed.
        if any(part in {"__pycache__", "outputs", "checkpoints"} for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".part"}:
            continue
        entries.append(path)
    for name in (
        "README.md",
        "report.tex",
        "report_cn.tex",
        "references.tex",
        "requirements-report.txt",
    ):
        path = root / name
        if path.exists():
            entries.append(path)
    for folder in ("tables", "figures"):
        entries.extend(p for p in (root / folder).glob("*") if p.is_file())
    entries = sorted(set(entries))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in entries:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr(
            "package_manifest.json",
            json.dumps(
                {
                    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in entries
                },
                indent=2,
            ),
        )
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    print(f"Created {output} ({output.stat().st_size:,} bytes; {len(entries)} files).")
    print("Upload report.pdf separately; report_cn.pdf is the Chinese companion.")


if __name__ == "__main__":
    main()
