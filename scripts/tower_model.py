import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 300

class TwoTowerModel(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()

        self.emb = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.proj_query = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.ReLU(),
            nn.LayerNorm(EMBED_DIM),
            nn.Dropout(0.1),
        )

        self.proj_doc = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.ReLU(),
            nn.LayerNorm(EMBED_DIM),
            nn.Dropout(0.1),
        )

    def masked_mean_pool(self, x, mask):
        x = x * mask.unsqueeze(-1)             # zero out padded values
        summed = x.sum(dim=1)                  # sum across sequence
        lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(1)  # (batch, 1)
        return summed / lengths                # mean without padding

    def encode(self, x, tower: str = "query"):
        """Encodes a batch of tokenized queries or passages."""
        mask = (x != 0).float()                # 1 for real tokens, 0 for PAD
        x = self.emb(x)                        # (batch, seq_len, emb_dim)
        x = self.masked_mean_pool(x, mask)     # (batch, emb_dim)

        if tower == "query":
            x = self.proj_query(x)
        else:
            x = self.proj_doc(x)

        return F.normalize(x, dim=1)           # unit-normalized for cosine

    def encode_query(self, q):
        return self.encode(q, tower="query")

    def encode_passage(self, p):
        return self.encode(p, tower="doc")

    def forward(self, q, p, n):
        q_enc = self.encode(q, tower="query")
        p_enc = self.encode(p, tower="doc")
        n_enc = self.encode(n, tower="doc")
        return q_enc, p_enc, n_enc