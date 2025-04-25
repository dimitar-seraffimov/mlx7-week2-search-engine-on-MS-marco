import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TripletMarginWithDistanceLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import datetime
from tower_model import TwoTowerModel
import wandb
from evaluate import evaluate, val_loader, mrr_at_k, ndcg_at_k
from chromadb import PersistentClient
from s02_tkn_ms_marco import text_to_ids

#
#
# SETUP
#
#

EPOCHS = 10
BATCH_SIZE = 128
EMBED_DIM = 300
LEARNING_RATE = 1e-3
MARGIN = 0.55
GRAD_CLIP = 1.0

CHECKPOINT_PATH = Path("../checkpoint_hard.pt")
EMBEDDING_MATRIX_PATH = Path("../embedding_matrix.npy")
TRIPLETS_PATH = Path("../train_tokenised_hard.parquet")

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

vocab_to_int = pd.read_parquet("../tkn_vocab_to_int.parquet").iloc[0].to_dict()

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
    def pad(seqs): return nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return {
        "query": pad([item["query"] for item in batch]),
        "pos": pad([item["pos"] for item in batch]),
        "neg": pad([item["neg"] for item in batch]),
    }

#
#
# TRAINING
#
#

def main():
    wandb.init(
        project="mlx7-week2-search-engine",
        name=f"hard-negatives-{timestamp}",
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "margin": MARGIN,
            "triplets_path": str(TRIPLETS_PATH),
            "embedding_dim": EMBED_DIM
        }
    )

    print("[Step 1] Loading embedding matrix...")
    embedding_matrix = torch.tensor(np.load(EMBEDDING_MATRIX_PATH), dtype=torch.float32)

    print("[Step 2] Loading hard-negative triplets...")
    df = pd.read_parquet(TRIPLETS_PATH)
    dataset = TripletDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print("[Step 3] Building model...")
    model = TwoTowerModel(embedding_matrix).to(device)
    wandb.watch(model, log="all") 
    print('Model parameters:', sum(p.numel() for p in model.parameters()))
    
    # define triplet loss function with Euclidean (L2) distance and triplet loss
    cosine_distance = lambda x, y: 1 - F.cosine_similarity(x, y, dim=1)
    criterion = TripletMarginWithDistanceLoss(
        distance_function=cosine_distance,
        margin=MARGIN
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    print("[Step 4] Training with HARD negatives...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for i, batch in enumerate(progress_bar):
            q = batch["query"].to(device)
            p = batch["pos"].to(device)
            n = batch["neg"].to(device)

            optimizer.zero_grad()
            # encode query, positive, and negative passages
            q_enc, p_enc, n_enc = model(q, p, n)

            loss = criterion(q_enc, p_enc, n_enc) # compute loss here
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # log after each batch
            wandb.log({"loss": loss.item(), "epoch": epoch + 1, "batch": i})
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        if val_loader:
            val_loss = evaluate(model, val_loader, criterion)
            print(f"[Val] Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
            wandb.log({"val_loss": val_loss, "epoch": epoch + 1})
            
            # connect to the pre-cached Chroma collection
            client = PersistentClient(path="../chromadb")
            collection = client.get_or_create_collection(
                name="document", metadata={"distance_metric":"cosine"}
            )

            # load the raw validation set
            val_df = pd.read_parquet("../validation_tokenised.parquet")

            ranked_lists, relevant_sets = [], []
            for row in val_df.itertuples():
                # encode the query text
                ids = text_to_ids(row.query, vocab_to_int)
                q_tensor = torch.tensor([ids], dtype=torch.long).to(device)
                with torch.no_grad():
                    q_vec = model.encode(q_tensor).squeeze(0).cpu().numpy()

                # retrieve top-10 from ChromaDB
                results = collection.query(
                    query_embeddings=[q_vec],
                    n_results=10
                )
                docs = results["documents"][0]     # list of passage texts
                ranked_lists.append(docs)
                relevant_sets.append({row.positive_passage})

            # 3) Compute metrics
            val_mrr  = mrr_at_k(ranked_lists, relevant_sets, k=10)
            val_ndcg = ndcg_at_k(ranked_lists, relevant_sets, k=10)
            print(f"[Val] Epoch {epoch+1} → MRR@10: {val_mrr:.4f}, nDCG@10: {val_ndcg:.4f}")
            wandb.log({"val_mrr": val_mrr, "val_ndcg": val_ndcg, "epoch": epoch+1})

        # save checkpoint for each epoch
        checkpoint_name = f"checkpoint_hard_{timestamp}_epoch_{epoch+1}.pt"
        checkpoint_path = Path(f"../checkpoints/{checkpoint_name}")
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        
        # log model to wandb
        artifact = wandb.Artifact(f'model-epoch-{epoch+1}', type='model')
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)

    # save final model
    print(f"[✓] Saving final checkpoint to {CHECKPOINT_PATH}")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    
    # finish wandb run
    wandb.finish()
    print("[DONE]")

if __name__ == "__main__":
    main()