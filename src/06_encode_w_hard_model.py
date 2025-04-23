import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tower_model import TwoTowerModel
import chromadb
from torch.utils.data import DataLoader
from chromadb import PersistentClient

#
#
# SETUP
#
#

EMBEDDING_MATRIX_PATH = Path("../embedding_matrix.npy")
CHECKPOINT_PATH = Path("../checkpoint_hard.pt")
TOKENISED_DATA_PATH = Path("../train_tokenised_hard.parquet")
CHROMA_COLLECTION_NAME = "document"
CHROMA_DB_DIR = "../chromadb"
BATCH_SIZE = 1024

#
#
# INITIALISE CHROMADB
#
#

chroma_client = PersistentClient(path=CHROMA_DB_DIR)

if CHROMA_COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection(CHROMA_COLLECTION_NAME)

collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

#
#
# ENCODE & ADD TO CHROMADB (BATCHED)
#
#

def encode_passages():
    print("[Step 1] Loading trained HARD model...")
    embedding_matrix = torch.tensor(np.load(EMBEDDING_MATRIX_PATH), dtype=torch.float32)
    model = TwoTowerModel(embedding_matrix)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    print("[Step 2] Loading tokenised data...")
    df = pd.read_parquet(TOKENISED_DATA_PATH)

    print("[Step 3] Encoding and adding to ChromaDB in batches...")
    with torch.no_grad():
        for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Encoding passages"):
            batch = df.iloc[i:i+BATCH_SIZE]

            passages = [torch.tensor(x, dtype=torch.long) for x in batch["pos_ids"]]
            passages_padded = nn.utils.rnn.pad_sequence(passages, batch_first=True, padding_value=0)

            embeddings = model.encode(passages_padded).numpy().tolist()

            doc_texts = list(batch["positive_passage"])
            doc_ids = [f"doc_{i + j}" for j in range(len(batch))]

            collection.add(
                documents=doc_texts,
                embeddings=embeddings,
                ids=doc_ids
            )

    print("[✓] Encoding complete. ChromaDB updated with HARD model vectors.")

#
#
# RUN
#
#

if __name__ == "__main__":
    encode_passages()
