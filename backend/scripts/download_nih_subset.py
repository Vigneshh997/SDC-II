import os
from urllib.parse import urlparse
import requests
import tarfile
from pathlib import Path
import time


def download_with_retry(url, dest, max_retries=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    if urlparse(url).scheme != "https":
        raise ValueError("Dataset downloads require HTTPS")

    for attempt in range(max_retries):
        try:
            with (
                requests.get(url, headers=headers, stream=True, timeout=60) as response,
                open(dest, "wb") as out_file,
            ):
                response.raise_for_status()
                # We do a basic block-by-block download
                block_size = 1024 * 1024 * 5  # 5MB blocks
                for data in response.iter_content(block_size):
                    out_file.write(data)
            return True
        except requests.HTTPError as e:
            print(f"HTTP download error on attempt {attempt + 1}: {type(e).__name__}")
            time.sleep(5)
        except Exception as e:
            print(f"Error {e} on attempt {attempt + 1}")
            time.sleep(5)

    print(f"Failed to download {url} after {max_retries} attempts.")
    return False


def download_subset():
    print("=" * 50)
    print("NIH Chest X-ray Subset Downloader (~10GB)")
    print("=" * 50)

    data_root = Path("backend/data/nih_cxr14")
    images_dir = data_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # We only download the first 4 archives to get ~10GB of data
    links = [
        "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
        "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
        "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2vd.gz",
        "https://nihcc.box.com/shared/static/0aowwzs5lhqvvcjpouqwmq5kasigws9m.gz",
    ]

    for idx, link in enumerate(links):
        fn = data_root / ("images_%02d.tar.gz" % (idx + 1))

        if not fn.exists() or fn.stat().st_size < 1000000:  # Less than 1MB is a failed download
            print(f"Downloading {fn.name} (~2.5GB)... This may take some time.")
            success = download_with_retry(link, fn)
            if not success:
                print("ABORTING: Could not download required files.")
                return
            print(f"Downloaded {fn.name}")
        else:
            print(f"{fn.name} already exists. Skipping download.")

        print(f"Extracting {fn.name}...")
        try:
            with tarfile.open(fn, "r:gz") as tar:
                # Filter members to only extract image files directly into images_dir
                for member in tar.getmembers():
                    if member.isreg() and member.name.endswith(".png"):
                        member.name = os.path.basename(member.name)
                        tar.extract(member, path=images_dir)
            print(f"Finished extracting {fn.name}")
        except EOFError:
            print(f"Archive {fn.name} is corrupted. Deleting it...")
            fn.unlink()

    print("\nDownloading Metadata (Data_Entry_2017_v2020.csv)...")
    csv_url = "https://nihcc.box.com/shared/static/qg9yffz35k80n9m10o4dpt48j72he2w8.csv"
    csv_path = data_root / "Data_Entry_2017_v2020.csv"
    download_with_retry(csv_url, csv_path)

    print("Downloading train/val and test split lists...")
    train_val_url = "https://nihcc.box.com/shared/static/86tbr5z96v0s57eay4l3x920u8s833f4.txt"
    test_url = "https://nihcc.box.com/shared/static/48pobq33b6807s727j8m0k4y97x3z1f6.txt"

    download_with_retry(train_val_url, data_root / "train_val_list.txt")
    download_with_retry(test_url, data_root / "test_list.txt")

    print("\n" + "=" * 50)
    print("Subset download and extraction complete!")
    print(f"Images are stored in {images_dir}")
    print("=" * 50)


if __name__ == "__main__":
    download_subset()
