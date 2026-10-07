"""Download a Kaggle dataset.

Token is read from .env (KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY),
or falls back to the default ~/.kaggle/kaggle.json.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --slug marawanxmamdouh/dialogsum
    uv run python scripts/download_data.py --out data/raw --force
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_SLUG = "marawanxmamdouh/dialogsum"
# Ожидаемые файлы после распаковки DialogSum (jsonl-версия с Kaggle).
EXPECTED_FILES = ("dialogsum.train.jsonl", "dialogsum.dev.jsonl", "dialogsum.test.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset.")
    parser.add_argument(
        "--slug",
        type=str,
        default=DEFAULT_SLUG,
        help=f"Kaggle dataset slug (default: {DEFAULT_SLUG}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        help="Directory to store raw data (default: data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist.",
    )
    return parser.parse_args()


def load_kaggle_env() -> None:
    """Load .env and pass credentials to the kaggle CLI.

    Works with either:
      - KAGGLE_API_TOKEN=KGAT_...
      - KAGGLE_USERNAME=... + KAGGLE_KEY=...
      - legacy ~/.kaggle/kaggle.json (no env needed)
    """
    load_dotenv()

    token = os.getenv("KAGGLE_API_TOKEN")
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    if username:
        os.environ["KAGGLE_USERNAME"] = username
    if key:
        os.environ["KAGGLE_KEY"] = key

    has_env = bool(token) or bool(username and key)
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()

    if not has_env and not has_file:
        sys.exit(
            "[download] Kaggle credentials not found.\n"
            "  Provide one of:\n"
            "    .env: KAGGLE_API_TOKEN=KGAT_...\n"
            "    .env: KAGGLE_USERNAME=... / KAGGLE_KEY=...\n"
            "    file: ~/.kaggle/kaggle.json\n"
            "  Token page: https://www.kaggle.com/settings -> API -> Create New Token"
        )


def ensure_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        sys.exit(
            "[download] kaggle CLI not found.\n"
            "  Install it: uv add kaggle"
        )


def already_downloaded(out_dir: Path) -> bool:
    return all((out_dir / f).exists() for f in EXPECTED_FILES)


def clean_output_dir(out_dir: Path) -> None:
    for name in EXPECTED_FILES:
        p = out_dir / name
        if p.exists():
            print(f"[download] removing old {p}")
            p.unlink()
    for archive in out_dir.glob("*.zip"):
        print(f"[download] removing old {archive}")
        archive.unlink()


def download(out_dir: Path, slug: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        slug,
        "-p",
        str(out_dir),
        "--unzip",
    ]
    print(f"[download] running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        sys.exit(
            f"[download] kaggle CLI failed with exit code {e.returncode}.\n"
            f"  Check: (1) token is valid, (2) slug '{slug}' is correct, "
            f"(3) you accepted dataset terms on kaggle.com (if required)."
        )

    # Если по какой-то причине --unzip не сработал, распакуем сами.
    archives = list(out_dir.glob("*.zip"))
    for archive in archives:
        print(f"[download] unzipping {archive.name}")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
        archive.unlink()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    print(f"[download] dataset: {args.slug}")
    print(f"[download] target:  {out_dir.resolve()}")

    if already_downloaded(out_dir) and not args.force:
        print(
            f"[download] files already exist in {out_dir}.\n"
            f"[download] nothing to do. Use --force to re-download."
        )
        return

    load_kaggle_env()
    ensure_kaggle_cli()

    if args.force:
        clean_output_dir(out_dir)

    download(out_dir, args.slug)

    print(f"[download] done. Contents of {out_dir.resolve()}:")
    for p in sorted(out_dir.iterdir()):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()