import torch
import torch.nn as nn
from pathlib import Path

EMBED_DIM = 300

class TwoTowerModel(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)

    def encode(self, x):
        x = self.emb(x)
        x = x.mean(dim=1)  # Avg pooling
        return self.proj(x)

    def forward(self, q, p, n):
        q_enc = self.encode(q)
        p_enc = self.encode(p)
        n_enc = self.encode(n)
        return q_enc, p_enc, n_enc 