# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from tower_model import TwoTowerModel

#
# CONFIG
#

EMBEDDING_MATRIX_PATH = Path("../embedding_matrix.npy")
CHECKPOINT_PATH = Path("../checkpoint_hard.pt")
VAL_PARQUET_PATH = Path("../validation_tokenised.parquet")
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# MAIN
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

if __name__ == "__main__":
    run_evaluation()
