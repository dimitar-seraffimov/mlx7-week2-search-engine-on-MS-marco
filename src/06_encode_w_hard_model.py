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
# SETUP
#
#

EMBEDDING_MATRIX_PATH = Path("../embedding_matrix.npy")
CHECKPOINT_PATH = Path("../checkpoint_hard.pt")
TOKENISED_DATA_PATH = Path("../train_tokenised_hard.parquet")
CHROMA_COLLECTION_NAME = "document"

#
#
# INITIALISE CHROMADB
#
#

chroma_client = chromadb.PersistentClient(path="../chromadb")

# reset collection (optional, only if you want to start fresh)
if CHROMA_COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection(CHROMA_COLLECTION_NAME)

collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

#
#
# LOAD MODEL
#
#

print("[Step 1] Loading trained HARD model...")
embedding_matrix = torch.tensor(np.load(EMBEDDING_MATRIX_PATH), dtype=torch.float32)
model = TwoTowerModel(embedding_matrix)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()

#
#
# LOAD DATA
#
#

print("[Step 2] Loading tokenised data...")
df = pd.read_pickle(TOKENISED_DATA_PATH)

#
#
# ENCODE & ADD TO CHROMADB
#
#

print("[Step 3] Encoding and saving to ChromaDB...")
with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding passages"):
        passage_ids = torch.tensor(row["pos_ids"], dtype=torch.long).unsqueeze(0)
        passage_embedding = model.encode(passage_ids).squeeze(0).numpy()

        doc_id = f"doc_{idx}"
        collection.add(
            documents=[row["positive_passage"]],
            embeddings=[passage_embedding],
            ids=[doc_id]
        )

print("[✓] Encoding complete. ChromaDB updated with HARD model vectors.")