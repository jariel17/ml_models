import os
import requests
from zipfile import ZipFile
from typing import List, Dict

# --- Configuration for each dataset ---
DATASETS: List[Dict] = [
    {
        "name": "us_pictures",
        "url": "https://zenodo.org/record/7711412/files/US_Pictures.zip?download=1",
        "type": "zip",
        "destination": "data/external/us_pictures.zip",
        "extract_to": "data/raw/us_pictures"
    },
    {
        "name": "tabular_data",
        "url": "https://zenodo.org/record/7711412/files/tabular_data.csv?download=1",
        "type": "file",
        "destination": "data/raw/tabular_data.csv"
    },
    {
        "name": "test_set",
        "url": "https://zenodo.org/record/7711412/files/test_set.csv?download=1",
        "type": "file",
        "destination": "data/raw/test_set.csv"
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
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[INFO] Extracted to: {extract_to}")

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
