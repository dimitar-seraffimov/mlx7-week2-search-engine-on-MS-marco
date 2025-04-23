import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tower_model import TwoTowerModel
import chromadb
from chromadb.config import Settings

#
#
# SETUP:
#
#

EMBEDDING_MATRIX_PATH = Path("../embedding_matrix.npy")
CHECKPOINT_PATH = Path("../checkpoint_early.pt")
TOKENISED_DATA_PATH = Path("../train_tokenised.parquet")
CHROMA_COLLECTION_NAME = "document"

#
#
# INITIALISE CHROMADB:
#
#

chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="../chromadb"
))

#
#
# ENCODE & ADD TO CHROMADB:
#
#

def encode_passages():
    print("[Step 1] Loading model and checkpoint...")
    embedding_matrix = torch.tensor(np.load(EMBEDDING_MATRIX_PATH), dtype=torch.float32)
    model = TwoTowerModel(embedding_matrix)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    print("[Step 2] Loading tokenised data...")
    df = pd.read_parquet(TOKENISED_DATA_PATH)

    print("[Step 3] Creating Chroma collection...")
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("[Step 4] Encoding and adding to ChromaDB...")
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding passages"):
            passage_ids = torch.tensor(row["pos_ids"], dtype=torch.long).unsqueeze(0)
            # Encode only the positive passage for now
            passage_embedding = model.encode(passage_ids).squeeze(0).numpy()

            doc_id = f"doc_{idx}"
            collection.add(
                documents=[row["positive_passage"]],
                embeddings=[passage_embedding],
                ids=[doc_id]
            )

    print("[✓] Encoding complete. Collection saved.")
    chroma_client.persist()

if __name__ == "__main__":
    encode_passages()
