"""Download and unpack MovieLens-1M into a raw data directory."""
import zipfile
from pathlib import Path

import requests

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
REQUIRED_FILES = ("ratings.dat", "movies.dat", "users.dat")


def download_ml1m(raw_dir: Path) -> Path:
    """Fetch ML-1M into `raw_dir` and return the folder holding the .dat files.

    Does nothing if the files are already there.
    """
    raw_dir = Path(raw_dir)
    data_dir = raw_dir / "ml-1m"
    if all((data_dir / name).exists() for name in REQUIRED_FILES):
        return data_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-1m.zip"
    print(f"Downloading {ML1M_URL} ...")
    resp = requests.get(ML1M_URL, timeout=60)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    missing = [n for n in REQUIRED_FILES if not (data_dir / n).exists()]
    if missing:
        raise RuntimeError(f"Archive did not contain expected files: {missing}")
    return data_dir
