import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tower_model import TwoTowerModel

#
#
# CONFIGURATION
#
#

EPOCHS = 3
BATCH_SIZE = 128
EMBED_DIM = 300
CHECKPOINT_PATH = Path("../checkpoint_early.pt")
EMBEDDING_MATRIX_PATH = Path("../embedding_matrix.npy")
TRIPLETS_PATH = Path("../train_tokenised.parquet")  # using random negatives

#
#
# DATASET
#
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
    def pad(seq_list):
        return nn.utils.rnn.pad_sequence(seq_list, batch_first=True, padding_value=0)
    return {
        "query": pad([item["query"] for item in batch]),
        "pos": pad([item["pos"] for item in batch]),
        "neg": pad([item["neg"] for item in batch]),
    }

#
#
# MODEL
#
#

# Model definition moved at tower_model.py

#
#
# TRAINING
#
#

def main():
    # load embedding matrix
    embedding_matrix = torch.tensor(np.load(EMBEDDING_MATRIX_PATH), dtype=torch.float32)
    
    print("[Step 1] Loading tokenised triplets...")
    df = pd.read_parquet(TRIPLETS_PATH)
    dataset = TripletDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print("[Step 2] Building model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(embedding_matrix).to(device)

    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("[Step 3] Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            q = batch["query"].to(device)
            p = batch["pos"].to(device)
            n = batch["neg"].to(device)

            q_enc, p_enc, n_enc = model(q, p, n)

            loss = criterion(q_enc, p_enc, n_enc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    print(f"[✓] Saving checkpoint to {CHECKPOINT_PATH}")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print("[DONE]")

if __name__ == "__main__":
    main()