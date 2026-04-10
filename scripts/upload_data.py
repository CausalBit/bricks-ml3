"""Download the MovieLens 25M dataset and upload it to the landing volume.

Downloads the ML-25M zip from GroupLens, extracts the 6 CSV files to disk,
and uploads them to /Volumes/{catalog}/bronze/landing/ using the Databricks
CLI (handles large files with chunked uploads).

Usage:
    python scripts/upload_data.py --target dev
    python scripts/upload_data.py --target prod --skip-download  # reuse cached zip

Idempotent — skips files that already exist in the volume (use --force to
re-upload).

Authentication uses the standard Databricks environment variables:
    DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import yaml
from databricks.sdk import WorkspaceClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ZIP_PATH = DATA_DIR / "ml-25m.zip"
EXTRACT_DIR = DATA_DIR / "ml-25m"

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

# The 6 CSV files the pipeline expects (maps zip entry name → upload name)
CSV_FILES = {
    "ml-25m/ratings.csv": "ratings.csv",
    "ml-25m/movies.csv": "movies.csv",
    "ml-25m/tags.csv": "tags.csv",
    "ml-25m/genome-scores.csv": "genome-scores.csv",
    "ml-25m/genome-tags.csv": "genome-tags.csv",
    "ml-25m/links.csv": "links.csv",
}


def load_bundle_config() -> dict:
    with open(PROJECT_ROOT / "databricks.yml") as f:
        return yaml.safe_load(f)


def resolve_catalog(config: dict, target: str) -> str:
    targets = config.get("targets", {})
    if target not in targets:
        print(f"ERROR: target '{target}' not found. Available: {list(targets.keys())}")
        sys.exit(1)
    catalog = targets[target].get("variables", {}).get("catalog")
    if not catalog:
        print(f"ERROR: no 'catalog' variable for target '{target}'")
        sys.exit(1)
    return catalog


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset():
    """Download the ML-25M zip if not already cached."""
    if ZIP_PATH.exists():
        size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
        print(f"  Using cached {ZIP_PATH.name} ({size_mb:.0f} MB)")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading ML-25M dataset (~250 MB)...")
    print(f"  URL: {DATASET_URL}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        mb = downloaded / (1024 * 1024)
        print(f"\r  {mb:.0f} MB ({pct:.0f}%)", end="", flush=True)

    urlretrieve(DATASET_URL, ZIP_PATH, reporthook=progress)
    print()
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"  Downloaded {size_mb:.0f} MB")


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

def extract_csvs():
    """Extract the 6 CSV files from the zip to disk."""
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for zip_entry in CSV_FILES:
            dest = EXTRACT_DIR / Path(zip_entry).name
            if dest.exists():
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  {dest.name:25s} already extracted ({size_mb:.0f} MB)")
                continue

            info = zf.getinfo(zip_entry)
            size_mb = info.file_size / (1024 * 1024)
            print(f"  {dest.name:25s} extracting ({size_mb:.0f} MB)...", end="", flush=True)
            zf.extract(zip_entry, DATA_DIR)
            print(" done")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_csvs(w: WorkspaceClient, catalog: str, force: bool):
    """Upload extracted CSVs to the landing volume using the Databricks CLI."""
    volume_path = f"/Volumes/{catalog}/bronze/landing"

    print(f"  Uploading to {volume_path}/")

    for zip_entry, upload_name in CSV_FILES.items():
        local_file = EXTRACT_DIR / Path(zip_entry).name
        dest = f"{volume_path}/{upload_name}"

        if not force:
            try:
                status = w.files.get_status(dest)
                size_mb = (status.content_length or 0) / (1024 * 1024)
                print(f"  {upload_name:25s} already exists ({size_mb:.0f} MB), skipping")
                continue
            except Exception:
                pass  # file doesn't exist, upload it

        size_mb = local_file.stat().st_size / (1024 * 1024)
        print(f"  {upload_name:25s} uploading ({size_mb:.0f} MB)...", end="", flush=True)

        result = subprocess.run(
            ["databricks", "fs", "cp", str(local_file), dest, "--overwrite"],
            capture_output=True, text=True,
        )

        if result.returncode != 0:
            print(f" FAILED")
            print(f"    stderr: {result.stderr.strip()}")
            sys.exit(1)

        print(" done")

    print(f"  All CSV files uploaded to {volume_path}/")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_upload(w: WorkspaceClient, catalog: str):
    """List files in the landing volume to confirm upload."""
    volume_path = f"/Volumes/{catalog}/bronze/landing"
    print(f"\n  Verifying {volume_path}/:")

    try:
        entries = list(w.files.list_directory_contents(volume_path))
        for entry in entries:
            if entry.name and entry.name.endswith(".csv"):
                size_mb = (entry.file_size or 0) / (1024 * 1024)
                print(f"    {entry.name:25s} {size_mb:>7.1f} MB")

        csv_count = sum(1 for e in entries if e.name and e.name.endswith(".csv"))
        if csv_count == len(CSV_FILES):
            print(f"\n  All {csv_count} CSV files present.")
        else:
            print(f"\n  WARNING: Expected {len(CSV_FILES)} files, found {csv_count}")
    except Exception as e:
        print(f"  WARNING: Could not verify: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and upload ML-25M data")
    parser.add_argument("--target", required=True, choices=["dev", "staging", "prod"],
                        help="DABs target environment")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use previously cached zip")
    parser.add_argument("--force", action="store_true",
                        help="Re-upload files even if they already exist")
    args = parser.parse_args()

    config = load_bundle_config()
    catalog = resolve_catalog(config, args.target)

    print(f"--- Data upload for target '{args.target}' ---")
    print(f"  Catalog: {catalog}")
    print(f"  Volume:  /Volumes/{catalog}/bronze/landing/")
    print()

    # Step 1: Download
    if not args.skip_download:
        print("[1/4] Download ML-25M dataset")
        download_dataset()
    else:
        print("[1/4] Download skipped (--skip-download)")
        if not ZIP_PATH.exists():
            print(f"  ERROR: {ZIP_PATH} not found. Run without --skip-download first.")
            sys.exit(1)
    print()

    # Step 2: Extract
    print("[2/4] Extract CSV files")
    extract_csvs()
    print()

    # Step 3: Upload
    print("[3/4] Upload CSVs to landing volume")
    w = WorkspaceClient()
    upload_csvs(w, catalog, force=args.force)
    print()

    # Step 4: Verify
    print("[4/4] Verify upload")
    verify_upload(w, catalog)

    print()
    print("Data upload complete.")


if __name__ == "__main__":
    main()
