import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tower_model import TwoTowerModel
import chromadb
from tkn_ms_marco import text_to_ids
import torch.nn as nn

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
BATCH_SIZE = 1024

#
#
# LOAD VOCAB & MODEL
#
#

print("[Step 1] Loading vocabulary...")
vocab_df = pd.read_parquet("../tkn_vocab_to_int.parquet")
vocab_to_int = vocab_df.iloc[0].to_dict()

print("[Step 2] Loading model...")
embedding_matrix = torch.tensor(np.load(EMBEDDING_MATRIX_PATH), dtype=torch.float32)
model = TwoTowerModel(embedding_matrix)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()

#
#
# LOAD DATA & CHROMADB
#
#

print("[Step 3] Loading tokenised data...")
df = pd.read_parquet(TOKENISED_DATA_PATH)

print("[Step 4] Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="../chromadb")
collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)

#
#
# HARD NEGATIVE MINING
#
#

hard_triplets = []

print("[Step 5] Mining hard negatives from ChromaDB...")
with torch.no_grad():
  for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Mining"):
      batch = df.iloc[i:i+BATCH_SIZE]
      queries = [torch.tensor(q, dtype=torch.long) for q in batch["query_ids"]]
      queries_padded = nn.utils.rnn.pad_sequence(queries, batch_first=True, padding_value=0)

      query_embeddings = model.encode(queries_padded).numpy()

      results = collection.query(
          query_embeddings=query_embeddings.tolist(),
          n_results=5
      )

      for j, row in enumerate(batch.itertuples()):
          gold_text = row.positive_passage
          retrieved_docs = results["documents"][j]
          retrieved_ids = results["ids"][j]

          # find one valid hard negative
          hard_negative_text = next(
              (doc for doc in retrieved_docs if doc.strip() != gold_text.strip()),
              None
          )

          if not hard_negative_text:
              continue  # skip if none found

          hard_neg_ids = text_to_ids(hard_negative_text, vocab_to_int)
          if not hard_neg_ids:
              continue

          hard_triplets.append({
              "query": row.query,
              "query_ids": row.query_ids,
              "positive_passage": row.positive_passage,
              "pos_ids": row.pos_ids,
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