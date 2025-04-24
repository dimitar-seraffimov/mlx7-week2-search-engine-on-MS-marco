import torch
import numpy as np
import pandas as pd
from tower_model import TwoTowerModel
from pathlib import Path
from s02_tkn_ms_marco import text_to_ids
import chromadb
from chromadb.config import Settings

#
#
# SETUP
#
#

VOCAB_PATH = Path("../tkn_vocab_to_int.parquet")
EMBEDDING_MATRIX_PATH = Path("../embedding_matrix.npy")
CHECKPOINT_PATH = Path("../checkpoint_hard.pt")
CHROMA_DB_DIR = Path("../chromadb")
CHROMA_COLLECTION_NAME = "document"

#
#
# LOAD VOCABULARY
#
#

vocab_to_int = pd.read_parquet(VOCAB_PATH)

#
#
# LOAD CHROMADB
#
#

print("[Step 4] Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)

#
#
# LOAD MODEL
#
#

embedding_matrix = torch.tensor(np.load(EMBEDDING_MATRIX_PATH), dtype=torch.float32)
model = TwoTowerModel(embedding_matrix)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()

# === Query helper ===
def embed_query(query: str) -> np.ndarray:
    ids = text_to_ids(query, vocab_to_int)
    if not ids:
        raise ValueError("Query tokens are too unknown or empty.")
    query_tensor = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        embedding = model.encode(query_tensor).squeeze(0).numpy()
    return embedding

#
#
# QUERY HELPER
#
#

def query_chromadb(query: str, k: int = 5):
    print(f"\nQuery: {query}")
    try:
        query_vec = embed_query(query)
    except ValueError as e:
        print(f"[!] Cannot embed query: {e}")
        return

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=k
    )

    retrieved_docs = results["documents"][0]
    distances = results.get("distances", [])[0]  # cosine distances

    print("\nTop Results:")
    for rank, (doc, dist) in enumerate(zip(retrieved_docs, distances)):
        similarity = 1 - dist  # since cosine distance = 1 - cosine similarity
        print(f"{rank+1}. [Similarity: {similarity:.4f}] {doc.strip()[:200]}...\n")

#
#
# RUN QUERY LOOP
#
#

if __name__ == "__main__":
    while True:
        q = input("Enter your query (or 'exit'): ")
        if q.lower() in ("exit", "quit"):
            break
        query_chromadb(q, k=5)
