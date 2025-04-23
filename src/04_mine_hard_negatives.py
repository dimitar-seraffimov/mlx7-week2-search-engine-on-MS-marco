import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tower_model import TwoTowerModel
import chromadb
from chromadb.config import Settings
from tkn_ms_marco import text_to_ids

#
#
# SETUP
#
#

EMBEDDING_MATRIX_PATH = Path("../embedding_matrix.npy")
CHECKPOINT_PATH = Path("../checkpoint_early.pt")
TOKENISED_DATA_PATH = Path("../train_tokenised.parquet")
OUTPUT_PATH = Path("../train_tokenised_hard.parquet")
CHROMA_COLLECTION_NAME = "document"

#
#
# INITIALISE CHROMADB
#
#

chroma_client = chromadb.PersistentClient(path="../chromadb")
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

#
#
# LOAD TOKENISED DATA
#
#

df = pd.read_parquet(TOKENISED_DATA_PATH)

#
#
# HARD NEGATIVE MINING
#
#

hard_triplets = []

print("[Step] Mining hard negatives from ChromaDB...")
with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        query_ids = torch.tensor(row["query_ids"], dtype=torch.long).unsqueeze(0)
        query_embedding = model.encode(query_ids).squeeze(0).numpy()

        # retrieve top-k most similar passages
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        retrieved_docs = results["documents"][0]
        retrieved_ids = results["ids"][0]

        # avoid gold positive
        gold_text = row["positive_passage"]

        # pick a hard negative that's NOT the positive
        hard_negative_text = None
        for doc, doc_id in zip(retrieved_docs, retrieved_ids):
            if doc.strip() != gold_text.strip():
                hard_negative_text = doc
                break

        if hard_negative_text is None:
            continue  # fallback: skip if nothing found

        # convert hard negative text to token IDs
        # (reuse the vocab and text_to_ids function from earlier)

        vocab_df = pd.read_parquet("../tkn_vocab_to_int.parquet")
        vocab_to_int = vocab_df.iloc[0].to_dict()

        hard_neg_ids = text_to_ids(hard_negative_text, vocab_to_int)

        # save new triplet
        hard_triplets.append({
            "query": row["query"],
            "query_ids": row["query_ids"],
            "positive_passage": row["positive_passage"],
            "pos_ids": row["pos_ids"],
            "negative_passage": hard_negative_text,
            "neg_ids": hard_neg_ids,
        })

#
#
# SAVE NEW DATASET
#
#

new_df = pd.DataFrame(hard_triplets)
new_df.to_pickle(OUTPUT_PATH)
print(f"[✓] Saved {len(new_df)} mined triplets to: {OUTPUT_PATH}")
