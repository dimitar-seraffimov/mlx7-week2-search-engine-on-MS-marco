import pandas as pd
import os
import requests
from pathlib import Path

#
#
# SETUP:
# Set Hugging Face Parquet URLs (v1.1)
#
#

HF_PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/train-00000-of-00001.parquet",
    "validation": "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/validation-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/test-00000-of-00001.parquet",
}

#
#
# DOWNLOAD:
#
#

def download_if_not_exists(url, path):
    path = Path(path)
    if path.exists():
        print(f"[SKIP] Already downloaded: {path}")
        return
    print(f"[Downloading] {url}")
    r = requests.get(url)
    path.write_bytes(r.content)

#
#
# DOWNLOAD & LOAD:
#
#

splits = {}
for name, url in HF_PARQUET_URLS.items():
    parquet_path = f"../{name}.parquet"
    download_if_not_exists(url, parquet_path)

    print(f"Loading {name} data from {parquet_path}")
    splits[name] = pd.read_parquet(parquet_path)

#
#
# COMBINE:
#
#

print("Combining datasets...")
combined_data = pd.concat(list(splits.values()), ignore_index=True)

#
#
# SAVE:
#
#

output_file = "../combined.parquet"
combined_data.to_parquet(output_file)

print(f"[DONE] Combined dataset saved to {output_file}")
print(f"Total records: {len(combined_data):,}")