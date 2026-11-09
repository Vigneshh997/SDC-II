# backend/data/download_dataset.py
"""Download the NIH ChestX-ray14 dataset — a research-grade benchmark.

Dataset: 112,120 frontal-view chest X-rays from 30,805 unique patients
         with 14 disease labels + patient age, gender, view position.
Source:  NIH Clinical Center (Wang et al., CVPR 2017)
Paper:   "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks"

Usage:
    python -m backend.data.download_dataset --output_dir ./data/nih_cxr14
    python -m backend.data.download_dataset --output_dir ./data/nih_cxr14 --batches 2  # download only 2 image batches (~20K images)

The full dataset is ~45 GB (12 image archives). For faster iteration you can
download a subset by specifying --batches N (each batch is ~4 GB, ~10K images).
"""

import argparse
import os
import tarfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
import requests

from tqdm import tqdm


# ---- NIH Box.com download links (official) ----
# These are the direct-download URLs from NIH Clinical Center's Box repository.
IMAGE_URLS = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",  # images_001
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pber4ta4ol.gz",  # images_002
    "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20telehl.gz",  # images_003
    "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",  # images_004
    "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfamk.gz",  # images_005
    "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",  # images_006
    "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",  # images_007
    "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",  # images_008
    "https://nihcc.box.com/shared/static/upyy3ml7qduez2esga2r7eezt6a3jfbb.gz",  # images_009
    "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",  # images_010
    "https://nihcc.box.com/shared/static/eeq4r0ber90t3sre10lk5ly2ekngrtdl.gz",  # images_011
    "https://nihcc.box.com/shared/static/canbt8wowuhl2ht7z3nb2qmzv9iax1wb.gz",  # images_012
]

# Metadata + split files (from HuggingFace mirror for reliability)
METADATA_URL = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/Data_Entry_2017_v2020.csv"
TRAIN_VAL_LIST_URL = (
    "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/train_val_list.txt"
)
TEST_LIST_URL = (
    "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/test_list.txt"
)


class _DownloadProgressBar(tqdm):
    """tqdm wrapper for urlretrieve progress hook."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_file(url: str, dest: str, desc: str = ""):
    """Download a file with progress bar. Skip if already exists."""
    if os.path.exists(dest):
        print(f"  [SKIP] {desc} already exists: {dest}")
        return
    print(f"  [DOWNLOAD] {desc} -> {dest}")
    if urlparse(url).scheme != "https":
        raise ValueError("Dataset downloads require HTTPS")
    try:
        with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                t.total = int(response.headers.get("content-length", 0)) or None
                with open(dest, "wb") as output:
                    for block in response.iter_content(1024 * 1024):
                        output.write(block)
                        t.update(len(block))
    except (URLError, requests.RequestException) as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        print(f"  Please download manually and place at: {dest}")


def _extract_archive(archive_path: str, extract_dir: str):
    """Extract .tar.gz or .gz archive."""
    print(f"  [EXTRACT] {archive_path} -> {extract_dir}")
    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir, filter="data")
    elif archive_path.endswith(".gz"):
        # NIH files are actually tar.gz even with .gz extension
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_dir, filter="data")
        except tarfile.ReadError:
            print(f"  [WARNING] Could not extract {archive_path} as tar.gz")


def download_nih_cxr14(output_dir: str, num_batches: int = 12):
    """Download and extract the NIH ChestX-ray14 dataset.

    Args:
        output_dir:  Root directory for the dataset.
        num_batches: Number of image batches to download (1-12, each ~10K images).
                     Use fewer for faster iteration during development.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  NIH ChestX-ray14 Dataset Downloader")
    print(f"  Target: {output_path}")
    print(f"  Image batches: {num_batches} / 12")
    print("=" * 60)

    # 1. Download metadata CSV
    print("\n[Step 1/3] Downloading metadata...")
    csv_dest = str(output_path / "Data_Entry_2017_v2020.csv")
    _download_file(METADATA_URL, csv_dest, "Data_Entry_2017_v2020.csv")

    # 2. Download train/test split files
    print("\n[Step 2/3] Downloading official train/test splits...")
    _download_file(TRAIN_VAL_LIST_URL, str(output_path / "train_val_list.txt"), "train_val_list.txt")
    _download_file(TEST_LIST_URL, str(output_path / "test_list.txt"), "test_list.txt")

    # 3. Download image archives
    num_batches = min(num_batches, len(IMAGE_URLS))
    print(f"\n[Step 3/3] Downloading {num_batches} image batches (~{num_batches * 4} GB)...")
    print("           Each batch contains ~10,000 images.")
    print("           Full dataset = 12 batches = 112,120 images.\n")

    archives_dir = output_path / "archives"
    archives_dir.mkdir(exist_ok=True)

    for i in range(num_batches):
        archive_name = f"images_{i + 1:03d}.tar.gz"
        archive_path = str(archives_dir / archive_name)

        # Check if images from this batch are already extracted
        _download_file(IMAGE_URLS[i], archive_path, archive_name)

        # Extract
        if os.path.exists(archive_path):
            _extract_archive(archive_path, str(output_path))

    # Count total images
    image_files = list(images_dir.glob("*.png"))
    print(f"\n{'=' * 60}")
    print("  Download complete!")
    print(f"  Total images: {len(image_files)}")
    print(f"  Metadata:     {csv_dest}")
    print(f"  Location:     {output_path}")
    print(f"{'=' * 60}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Download NIH ChestX-ray14 (research-grade benchmark)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/nih_cxr14",
        help="Root directory for dataset (default: ./data/nih_cxr14)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=2,
        help="Number of image batches to download, 1-12 (default: 2 = ~20K images)",
    )
    args = parser.parse_args()
    download_nih_cxr14(args.output_dir, num_batches=args.batches)


if __name__ == "__main__":
    main()
