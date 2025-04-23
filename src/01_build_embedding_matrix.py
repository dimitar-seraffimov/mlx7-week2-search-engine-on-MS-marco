import numpy as np
import pandas as pd
from gensim.downloader import load
from pathlib import Path

#
# 
# STEP 1: SET PATHS
#
#

script_dir = Path(__file__).parent
root_dir = script_dir.parent if script_dir.name == "src" else script_dir
vocab_path = root_dir / "tkn_vocab_to_int.parquet"
output_path = root_dir / "embedding_matrix.npy"

#
#
# STEP 2: LOAD VOCAB MAPPINGS
#
#

print("[Step 1] Loading vocab...")
df_vocab = pd.read_parquet(vocab_path)
vocab_to_int = df_vocab.iloc[0].to_dict()
#
#
# STEP 3: Load pretrained word2vec model
#
#

print("[Step 2] Loading pretrained embeddings from Hugging Face...")

# Options include:
#   "glove-wiki-gigaword-300"
#   "word2vec-google-news-300"
#   "fasttext-wiki-news-subwords-300"
embedding_model_name = "glove-wiki-gigaword-300"
word2vec = load(embedding_model_name)

#
#
# STEP 3: Create embedding matrix aligned with vocab
#
#

print("[Step 3] Building embedding matrix...")

dim = word2vec.vector_size
unk_vector = np.random.normal(0, 1, dim)
embedding_matrix = np.zeros((len(vocab_to_int), dim))

found, missing = 0, 0

for word, idx in vocab_to_int.items():
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]
        found += 1
    else:
        embedding_matrix[idx] = unk_vector
        missing += 1

print(f"[INFO] Found {found} / {len(vocab_to_int)} words in the pretrained embeddings")
print(f"[INFO] Missing {missing} words assigned random vectors")

#
#
# STEP 4: Save final matrix for torch.nn.Embedding.from_pretrained()
#
#

print("[Step 4] Saving embedding_matrix.npy...")

np.save(output_path, embedding_matrix)
print(f"[DONE] embedding_matrix.npy saved to: {output_path}")
