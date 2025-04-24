import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tower_model import TwoTowerModel
import chromadb
import wandb
import shutil

#
# WANDB CHECKPOINT DOWNLOAD
#

WANDB_USER = "mlx7-dimitar-projects"
WANDB_PROJECT = "mlx7-week2-search-engine"
ARTIFACT_NAME = "model-epoch-10:latest"
LOCAL_CHECKPOINT = Path("../checkpoint_hard.pt")

# DOWNLOAD CHECKPOINT
print("[W&B] Logging in and fetching checkpoint...")
wandb.login()
wandb.init(project=WANDB_PROJECT, name="download-hard-model", job_type="download")

artifact = wandb.use_artifact(f"{WANDB_USER}/{WANDB_PROJECT}/{ARTIFACT_NAME}", type="model")
artifact_dir = artifact.download()
# RENAME FILE TO checkpoint_hard.pt
downloaded = list(Path(artifact_dir).glob("*.pt"))[0]
shutil.copy(downloaded, LOCAL_CHECKPOINT)
print(f"[✓] Downloaded and saved checkpoint to {LOCAL_CHECKPOINT}")

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
NUM_WORKERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
#
# INITIALISE CHROMADB
#
#

print("[Step 4] Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME, 
    metadata={"distance_metric": "cosine"}
)


#
#
# DATASET + LOADER
#
#

class PassageDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "ids": f"doc_{idx}",
            "text": self.df.iloc[idx]["positive_passage"],
            "input_ids": torch.tensor(self.df.iloc[idx]["pos_ids"], dtype=torch.long)
        }

def collate_fn(batch):
    padded_inputs = nn.utils.rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    texts = [b["text"] for b in batch]
    ids = [b["ids"] for b in batch]
    return {"input_ids": padded_inputs, "texts": texts, "ids": ids}


#
#
# ENCODE & ADD TO CHROMADB (BATCHED)
#
#
def encode_passages():
    print("[Step 1] Loading trained HARD model...")
    embedding_matrix = torch.tensor(np.load(EMBEDDING_MATRIX_PATH), dtype=torch.float32)
    model = TwoTowerModel(embedding_matrix)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()

    print("[Step 2] Loading tokenised data...")
    df = pd.read_parquet(TOKENISED_DATA_PATH)
    dataset = PassageDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    print("[INFO] Clearing existing ChromaDB documents...")
    print("[DEBUG] Sample docs before deletion:", collection.peek())
    all_ids = collection.get()["ids"]
    for i in tqdm(range(0, len(all_ids), 500), desc="Deleting"):
        collection.delete(ids=all_ids[i:i + 500])
    print("[DEBUG] Deletion complete. Remaining docs:", collection.count())

    print("[Step 3] Encoding and batching for ChromaDB...")
    buffer_ids, buffer_docs, buffer_embs = [], [], []
    FLUSH_SIZE = 1024

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding passages"):
            inputs = batch["input_ids"].to(device)
            embeddings = model.encode(inputs).cpu().numpy().tolist()
            buffer_ids.extend(batch["ids"])
            buffer_docs.extend(batch["texts"])
            buffer_embs.extend(embeddings)

            if len(buffer_ids) >= FLUSH_SIZE:
                collection.add(documents=buffer_docs, embeddings=buffer_embs, ids=buffer_ids)
                buffer_ids, buffer_docs, buffer_embs = [], [], []

        # Final flush
        if buffer_ids:
            collection.add(documents=buffer_docs, embeddings=buffer_embs, ids=buffer_ids)

    print("[✓] Encoding complete. ChromaDB updated with HARD model vectors.")

#
#
# RUN
#
#

if __name__ == "__main__":
    encode_passages()
