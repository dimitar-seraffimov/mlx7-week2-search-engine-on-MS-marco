#!/usr/bin/env python3
import os
from huggingface_hub import HfApi, upload_folder

REPO_ID  = "madnexx/mlx7-week2-artifacts"

# create the repo (ignore error if it already exists)
api = HfApi()
try:
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=True,
        token=HF_TOKEN
    )
    print(f"✅ Created repo {REPO_ID}")
except Exception as e:
    print(f"➡️  Repo might already exist: {e}")

# upload the folder containing the artifacts
#      tkn_vocab_to_int.parquet
#      embedding_matrix.npy
#      checkpoint_hard.pt
#      chromadb/   <- the entire ChromaDB folder
upload_folder(
    repo_id=REPO_ID,
    repo_type="dataset",
    folder_path=".",
    path_in_repo="",
    token=HF_TOKEN,
    repo_type="dataset",
    ignore_patterns=["*.py","*.ipynb"]  # optional, exclude code if you like
)
print("✅ Uploaded artifacts to the Hub!")
