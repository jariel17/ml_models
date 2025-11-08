import os
import requests
from zipfile import ZipFile, BadZipFile
from typing import List, Dict

# --- Configuration for each dataset ---
DATASETS: List[Dict] = [
    {
        "name": "appendicitis_pictures",
        "url": "https://zenodo.org/record/7711412/files/US_Pictures.zip?download=1",
        "type": "zip",
        "destination": "../external/appendicitis_pictures.zip",
        "extract_to": "../raw/appendicitis_pictures"
    },
    {
        "name": "appendicitis_tabular_data",
        "url": "https://zenodo.org/record/7711412/files/app_data.xlsx?download=1",
        "type": "file",
        "destination": "../raw/appendicitis_tabular_data.xlsx"
    },
    {
        "name": "appendicitis_test_set",
        "url": "https://zenodo.org/record/7711412/files/test_set_codes.csv?download=1",
        "type": "file",
        "destination": "../raw/appendicitis_test_set.csv"
    }
]

# --- Core utilities ---
def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"[INFO] File already exists: {dest_path}")
        return
    print(f"[INFO] Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[INFO] Download completed: {dest_path}")

def extract_zip(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    try:
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[INFO] Extracted to: {extract_to}")
    except BadZipFile:
        print(f"[WARNING] Skipping extraction: {zip_path} is not a valid ZIP file.")

def handle_dataset(config):
    url = config["url"]
    dest = config["destination"]
    download_file(url, dest)

    if config["type"] == "zip":
        extract_to = config.get("extract_to", os.path.splitext(dest)[0])
        extract_zip(dest, extract_to)

# --- Main script ---
if __name__ == "__main__":
    for ds in DATASETS:
        print(f"\n=== Processing dataset: {ds['name']} ===")
        handle_dataset(ds)
