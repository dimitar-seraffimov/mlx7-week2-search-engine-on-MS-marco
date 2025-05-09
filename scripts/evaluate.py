# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from tower_model import TwoTowerModel
import numpy as np
from chromadb import PersistentClient
from s02_tkn_ms_marco import text_to_ids

#
#
# CONFIG
#
#

TOP_K = 10
EMBEDDING_MATRIX_PATH = Path("../embedding_matrix.npy")
CHECKPOINT_PATH = Path("../checkpoint_hard.pt")
VAL_PARQUET_PATH = Path("../validation_tokenised.parquet")
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_to_int = pd.read_parquet("../tkn_vocab_to_int.parquet").iloc[0].to_dict()


#
# Dataset & Collate
#

class TripletDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "query": torch.tensor(row["query_ids"], dtype=torch.long),
            "pos": torch.tensor(row["pos_ids"], dtype=torch.long),
            "neg": torch.tensor(row["neg_ids"], dtype=torch.long),
        }

def collate_fn(batch):
    def pad(seqs): return nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return {
        "query": pad([item["query"] for item in batch]),
        "pos": pad([item["pos"] for item in batch]),
        "neg": pad([item["neg"] for item in batch]),
    }

# Load validation data at module level for import
print("[Loading validation data for evaluation...]")
df = pd.read_parquet(VAL_PARQUET_PATH)
dataset = TripletDataset(df)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

#
# EVALUATION FUNCTION
#

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            q = batch["query"].to(DEVICE)
            p = batch["pos"].to(DEVICE)
            n = batch["neg"].to(DEVICE)

            q_enc, p_enc, n_enc = model(q, p, n)
            loss = criterion(q_enc, p_enc, n_enc)
            total_loss += loss.item()
    return total_loss / len(val_loader)

#
#
# RETRIEVAL EVALUATION
#
#

def mrr_at_k(ranked_lists, relevant_sets, k):
    rr_scores = []
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        rr = 0.0
        for idx, doc_id in enumerate(ranked[:k]):
            if doc_id in relevant:
                rr = 1.0 / (idx + 1)
                break
        rr_scores.append(rr)
    return np.mean(rr_scores)

def ndcg_at_k(ranked_lists, relevant_sets, k):
    def dcg(scores):
        return sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(scores))
    ndcg_scores = []
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        gains = [1 if doc_id in relevant else 0 for doc_id in ranked[:k]]
        ideal = sorted(gains, reverse=True)
        ndcg_scores.append(dcg(gains) / dcg(ideal) if sum(ideal) > 0 else 0)
    return np.mean(ndcg_scores)

#
#
# MAIN
#
#

def run_evaluation():
    print("[Step 1] Loading embedding matrix and validation data...")
    embedding_matrix = torch.tensor(
        pd.read_parquet(EMBEDDING_MATRIX_PATH).values, dtype=torch.float32
    )
    
    print("[Step 2] Loading model checkpoint...")
    model = TwoTowerModel(embedding_matrix).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    criterion = nn.TripletMarginLoss(margin=0.2, p=2)

    print("[Step 3] Evaluating...")
    avg_loss = evaluate(model, val_loader, criterion)
    print(f"[✓] Validation Loss: {avg_loss:.4f}")

    #
    # RETRIEVAL EVALUATION
    #

    print("[Step 4] Retrieval Evaluation as MRR@K and NDCG@K...")
    client = PersistentClient(path="../chromadb")
    collection = client.get_or_create_collection(
        name="document", metadata={"distance_metric":"cosine"}
    )

    # — build ranked_lists & relevant_sets
    val_df = pd.read_parquet(VAL_PARQUET_PATH)
    val_df = val_df.reset_index().rename(columns={"index":"id"})


    ranked_lists, relevant_sets = [], []
    for row in val_df.itertuples():
        # embed query
        ids = text_to_ids(row.query, vocab_to_int) 
        q_tensor = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            q_vec = model.encode(q_tensor).squeeze(0).cpu().numpy()

        # retrieve top-K from ChromaDB
        results = collection.query(
            query_embeddings=[q_vec],
            n_results=TOP_K
        )
        doc_ids = results["ids"][0]       # <-- list of retrieved IDs
        ranked_lists.append(doc_ids)
        relevant_sets.append({row.id})    # <-- compare to the ground-truth ID

    # — compute and display metrics
    mrr  = mrr_at_k(ranked_lists, relevant_sets, k=TOP_K)
    ndcg = ndcg_at_k(ranked_lists, relevant_sets, k=TOP_K)
    print(f"MRR@{TOP_K}:  {mrr:.4f}")
    print(f"nDCG@{TOP_K}: {ndcg:.4f}")





if __name__ == "__main__":
    run_evaluation()
