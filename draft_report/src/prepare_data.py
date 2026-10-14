"""Download the original Monash releases; select frozen IDs, never synthetic data."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from config import DATA_DIR, ELECTRICITY_CONFIG, SOLAR_CONFIG

RELEASES = {
    "Electricity": ("4656140", "electricity_hourly_dataset.zip", "321"),
    "Solar": ("4656144", "solar_10_minutes_dataset.zip", "137"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsf(path, channel_ids):
    """Read exact series_name fields, preserving source order and timestamps."""
    selected, means, names, lengths, starts = {}, [], [], [], []
    active = False
    with open(path, encoding="cp1252") as f:
        for line in f:
            line = line.strip()
            if line.lower() == "@data":
                active = True
                continue
            if not active or not line or line.startswith("#"):
                continue
            name, start, values = line.split(":", 2)
            a = np.fromstring(values, sep=",", dtype=np.float64)
            if not len(a) or not np.isfinite(a).all() or "?" in values:
                raise ValueError(f"Invalid source series {name}")
            b1 = int(0.7 * len(a))
            if a[:b1].std() <= 1e-8:
                raise ValueError(f"Constant training channel: {name}")
            names.append(name)
            means.append(float(a[:b1].mean()))
            lengths.append(len(a))
            starts.append(start)
            if name in channel_ids:
                selected[name] = a.astype(np.float32)
    # Independently reproduce the outline's training-only stratified selection.
    order = np.argsort(means, kind="stable")
    rng = np.random.default_rng(42)
    chosen = set(
        int(i) for group in np.array_split(order, 4) for i in rng.choice(group, 4, replace=False)
    )
    reproduced = [n for i, n in enumerate(names) if i in chosen]
    if reproduced != channel_ids:
        raise ValueError(f"Frozen ID mismatch: {reproduced} != {channel_ids}")
    if len(set(lengths)) != 1 or len(set(starts)) != 1:
        raise ValueError("Source series are not aligned")
    return np.column_stack([selected[n] for n in channel_ids]), {
        "all_series_count": len(names),
        "all_series_length": lengths[0],
        "start_timestamp": starts[0],
        "channel_ids": channel_ids,
        "selection_reproduced": reproduced,
        "all_finite": True,
    }


def prepare():
    root = Path(DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    for cfg in (ELECTRICITY_CONFIG, SOLAR_CONFIG):
        record, filename, count = RELEASES[cfg.name]
        url = f"https://zenodo.org/records/{record}/files/{filename}?download=1"
        archive = root / filename
        if not archive.exists():
            print("Downloading", url, flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "LDI-course-project/1.0"})
            with urllib.request.urlopen(request, timeout=120) as r, open(
                str(archive) + ".part", "wb"
            ) as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            Path(str(archive) + ".part").replace(archive)
        with zipfile.ZipFile(archive) as z:
            member = next(n for n in z.namelist() if n.endswith(".tsf"))
            # Extract only the expected data file, without trusting archive paths.
            tsf = root / Path(member).name
            tsf.write_bytes(z.read(member))
        matrix, metadata = read_tsf(tsf, cfg.channel_ids)
        if matrix.shape != (cfg.total_length, cfg.num_channels):
            raise ValueError(f"Unexpected data shape {matrix.shape}")
        if metadata["all_series_count"] != int(count):
            raise ValueError("Unexpected source channel count")
        cached = root / f"{cfg.name}_16ch.npy"
        np.save(cached, matrix)
        metadata.update(
            source_url=url,
            archive_sha256=sha256(archive),
            tsf_sha256=sha256(tsf),
            cache_sha256=sha256(cached),
            dtype=str(matrix.dtype),
            source_kind="Monash original release",
            unit="original release units; no unit conversion",
            dataset=cfg.name,
        )
        (root / f"{cfg.name}_manifest.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(cfg.name, matrix.shape, metadata["cache_sha256"], flush=True)


if __name__ == "__main__":
    prepare()
