import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tower_model import TwoTowerModel

def test_two_tower_model_shapes():
    vocab_size = 10
    embedding_dim = 300
    batch_size = 4
    seq_len = 6

    # Dummy embedding matrix
    embedding_matrix = torch.randn(vocab_size, embedding_dim)

    # Model
    model = TwoTowerModel(embedding_matrix)
    model.eval()

    # Simulated batch
    q = torch.randint(0, vocab_size, (batch_size, seq_len))
    p = torch.randint(0, vocab_size, (batch_size, seq_len))
    n = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        q_enc, p_enc, n_enc = model(q, p, n)

    # Shape checks
    assert q_enc.shape == (batch_size, embedding_dim)
    assert p_enc.shape == (batch_size, embedding_dim)
    assert n_enc.shape == (batch_size, embedding_dim)
