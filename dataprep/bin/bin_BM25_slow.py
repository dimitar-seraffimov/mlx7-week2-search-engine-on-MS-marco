import pandas as pd
import pickle
from rank_bm25 import BM25Okapi
from pathlib import Path
from tqdm import tqdm

#
# CONFIGURATION
#
TRAIN_PICKLE = Path("../train_tokenised.pkl")
VOCAB_INT_TO_STR = Path("../tkn_int_to_vocab.pkl")
VOCAB_STR_TO_INT = Path("../tkn_vocab_to_int.pkl")
TRIPLET_PICKLE = Path("../train_triplets_bm25.pkl")

#
#
# Load tokenised training data
#
#

print("[Step 1] Loading tokenised training data...")
df = pd.read_pickle(TRAIN_PICKLE)
int_to_vocab = pickle.load(open(VOCAB_INT_TO_STR, "rb"))
vocab_to_int = pickle.load(open(VOCAB_STR_TO_INT, "rb"))

#
#
# Convert token IDs back to words
#
#

print("[Step 2] Decoding passages back to word lists...")
df["query_words"] = df["query_ids"].apply(lambda ids: [int_to_vocab.get(i, "<UNK>") for i in ids])
df["pos_words"] = df["pos_ids"].apply(lambda ids: [int_to_vocab.get(i, "<UNK>") for i in ids])
df["neg_words"] = df["neg_ids"].apply(lambda ids: [int_to_vocab.get(i, "<UNK>") for i in ids])

#
#
# Build BM25 corpus index (positive + negative passages)
#
#

print("[Step 3] Building BM25 index...")
corpus = df["pos_words"].tolist() + df["neg_words"].tolist()
bm25 = BM25Okapi(corpus)

#
#
# For each query, select hard negatives using BM25
#
#

print("[Step 4] Generating BM25-ranked triplets...")

triplets = []
unk_id = vocab_to_int["<UNK>"]

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Ranking negatives"):
    query_tokens = row["query_words"]
    pos_text = row["pos_words"]

    top = bm25.get_top_n(query_tokens, corpus, n=5)
    negatives = [doc for doc in top if doc != pos_text][:2]

    if negatives:
        neg_ids_list = []
        for neg in negatives:
            ids = [vocab_to_int.get(tok, unk_id) for tok in neg]
            if ids:
                neg_ids_list.append(ids)

        if neg_ids_list:
            triplets.append({
                "query_ids": row["query_ids"],
                "pos_ids": row["pos_ids"],
                "neg_ids": neg_ids_list,
            })

#
#
# Save the resulting triplets
#
#

print(f"[Step 5] Saving triplets to {TRIPLET_PICKLE}")
pd.DataFrame(triplets).to_pickle(TRIPLET_PICKLE)
print("[DONE] BM25 triplet file saved.")