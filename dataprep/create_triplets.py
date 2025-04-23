import pandas as pd
import random
import numpy as np
from pathlib import Path

def build_pools(df: pd.DataFrame):
    pos_map = {}
    neg_list = []
    for row in df.itertuples():
        qid, qtxt = row.query_id, row.query
        sel_arr = row.passages['is_selected']
        txt_arr = row.passages['passage_text']
        for sel, txt in zip(sel_arr, txt_arr):
            if not isinstance(txt, str) or not txt.strip():
                continue
            if sel == 1:
                pos_map.setdefault(qid, {"query": qtxt, "positives": []})["positives"].append(txt)
            else:
                neg_list.append((qid, txt))
    return pos_map, neg_list

def extract_triplets(df: pd.DataFrame, offset: int = 4) -> pd.DataFrame:
    triples = []
    n = len(df)
    for i in range(n):
        row = df.iloc[i]
        passages = row['passages']
        if not isinstance(passages, dict):
            continue
        sel = passages.get('is_selected', [])
        texts = passages.get('passage_text', [])
        if not sel or not texts:
            continue
        pos_idxs = np.where(np.array(sel) == 1)[0]
        if pos_idxs.size == 0:
            continue
        positive = texts[pos_idxs[0]]
        neg_row = df.iloc[(i + offset) % n]
        neg_passages = neg_row['passages']
        if not isinstance(neg_passages, dict):
            continue
        neg_texts = neg_passages.get('passage_text', [])
        if not neg_texts:
            continue
        negative = None
        for _ in range(3):
            candidate = random.choice(neg_texts)
            if candidate != positive:
                negative = candidate
                break
        if not negative:
            continue
        triples.append({
            'query': row['query'],
            'positive_passage': positive,
            'negative_passage': negative
        })
    return pd.DataFrame(triples)

def main():
    random.seed(42)
    combined_path = "../combined.parquet"
    df = pd.read_parquet(combined_path)
    print(f"Loaded {len(df)} queries.")

    pos_map, neg_list = build_pools(df)
    print(f"Queries total: {len(df)}")
    print(f"Queries with ≥1 positive: {len(pos_map)}")
    print(f"Negatives in pool: {len(neg_list)}")

    triples_df = extract_triplets(df, offset=4)
    print(f"Total triplets: {len(triples_df)}")

    # sample display
    sample = triples_df[['query', 'positive_passage', 'negative_passage']] \
        .sample(10, random_state=42) \
        .reset_index(drop=True)
    print("\nSample triplets:\n", sample.to_string(index=False))

    # shuffle and split
    shuffled = triples_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_df = shuffled.iloc[:n_train]
    val_df = shuffled.iloc[n_train:n_train+n_val]
    test_df = shuffled.iloc[n_train+n_val:]

    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    train_df.to_parquet("../train.parquet", index=False)
    val_df.to_parquet("../validation.parquet", index=False)
    test_df.to_parquet("../test.parquet", index=False)

if __name__ == '__main__':
    main()
