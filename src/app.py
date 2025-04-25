import streamlit as st
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tower_model import TwoTowerModel
from s02_tkn_ms_marco import text_to_ids
import chromadb

#
#
# SETUP
#
#

BASE = Path(__file__).parent.parent  # one level up from src/
defaults = {
    "VOCAB_PATH": str(BASE / "tkn_vocab_to_int.parquet"),
    "EMBEDDING_MATRIX_PATH": str(BASE / "embedding_matrix.npy"),
    "CHECKPOINT_PATH": str(BASE / "checkpoint_hard.pt"),
    "CHROMA_DB_DIR": str(BASE / "chromadb"),
    "CHROMA_COLLECTION_NAME": "document"
}



#
#
# LOAD COMPONENTS
#
#

@st.cache_resource
def load_components():
    vocab_to_int = pd.read_parquet(defaults['VOCAB_PATH']).iloc[0].to_dict()
    embedding_matrix = torch.tensor(np.load(defaults['EMBEDDING_MATRIX_PATH']), dtype=torch.float32)
    model = TwoTowerModel(embedding_matrix)
    model.load_state_dict(torch.load(defaults['CHECKPOINT_PATH'], map_location='cpu'))
    model.eval()
    chroma_client = chromadb.PersistentClient(path=defaults['CHROMA_DB_DIR'])
    collection = chroma_client.get_or_create_collection(
        name=defaults['CHROMA_COLLECTION_NAME'],
        metadata={"distance_metric": "cosine"}
    )
    return vocab_to_int, model, collection

#
#
# EMBED QUERY
#
#

@st.cache_data
def embed_query(query, vocab_to_int, model):
    ids = text_to_ids(query, vocab_to_int)
    if not isinstance(ids, list) or len(ids) == 0:
        return None
    tensor = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        return model.encode(tensor).squeeze(0).cpu().numpy()

#
#
# MAIN STREAMLIT UI
#
#

st.title("Two-Tower Search Engine")

vocab_to_int, model, collection = load_components()

query = st.text_input("Enter your query:")
k = st.slider("Number of results (k):", min_value=1, max_value=10, value=3)

if st.button("Search") and query:
    q_vec = embed_query(query, vocab_to_int, model)
    if q_vec is None:
        st.error("Unable to tokenize query; please try different text.")
    else:
        results = collection.query(query_embeddings=[q_vec.tolist()], n_results=k)
        docs = results['documents'][0]
        dists = results.get('distances', [[]])[0]
        similarities = [round(1 - d, 4) for d in dists]
        for i, (doc, sim) in enumerate(zip(docs, similarities), start=1):
            st.subheader(f"Result {i} — Similarity: {sim}")
            st.write(doc)
