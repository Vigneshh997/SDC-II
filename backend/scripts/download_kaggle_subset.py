import os
import subprocess
import shutil
import zipfile
from pathlib import Path


def download_kaggle_subset():
    print("=" * 50)
    print("NIH Chest X-ray Subset Downloader (Kaggle - ~11GB)")
    print("=" * 50)

    data_root = Path("backend/data/nih_cxr14")
    images_dir = data_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    kaggle_cmd = os.path.join("venv", "Scripts", "kaggle.exe")
    if not os.path.exists(kaggle_cmd):
        kaggle_cmd = "kaggle"

    try:
        # 1. Download the 11GB authentic subset of images (without --unzip to prevent MemoryError)
        print("Downloading 11GB image subset via Kaggle...")
        zip_path = data_root / "nih-chest-xray-13-labels.zip"

        # We will download the zip file to disk. Kaggle CLI will resume if partially downloaded.
        subprocess.run(
            [
                kaggle_cmd,
                "datasets",
                "download",
                "-d",
                "danielmadmon/nih-chest-xray-13-labels",
                "-p",
                str(data_root),
            ],
            check=True,
        )

        # 2. Extract manually using zipfile to avoid MemoryError
        print("\nExtracting dataset (this might take a few minutes)...")
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_root)
            # Delete the zip file to save space
            print("Extraction complete. Deleting zip file...")
            os.remove(zip_path)

        # 3. Re-organize the images into the original format (move from Classes/* to images/)
        print("\nRe-organizing images into original NIH format...")
        classes_dir = data_root / "Classes"
        if classes_dir.exists():
            for root, _, files in os.walk(classes_dir):
                for file in files:
                    if file.endswith(".png"):
                        src = Path(root) / file
                        dst = images_dir / file
                        if not dst.exists():
                            shutil.move(str(src), str(dst))
            shutil.rmtree(classes_dir)

        # 4. Download the original metadata
        print("\nDownloading official NIH metadata...")
        meta_files = ["Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt"]
        for file in meta_files:
            if not (data_root / file).exists():
                subprocess.run(
                    [
                        kaggle_cmd,
                        "datasets",
                        "download",
                        "-d",
                        "nih-chest-xrays/data",
                        "-f",
                        file,
                        "-p",
                        str(data_root),
                        "--unzip",
                    ],
                    check=True,
                )

        # Rename CSV to match project expectations
        src_csv = data_root / "Data_Entry_2017.csv"
        dst_csv = data_root / "Data_Entry_2017_v2020.csv"
        if src_csv.exists() and not dst_csv.exists():
            shutil.copy(src_csv, dst_csv)

        print("\n" + "=" * 50)
        print("Subset download and extraction complete!")
        print(f"Images are stored in {images_dir}")
        print("=" * 50)

    except subprocess.CalledProcessError as e:
        print("\nError running Kaggle command. Please ensure kaggle.json is in ~/.kaggle/!")
        print(e)
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    download_kaggle_subset()
