import pandas as pd
import random
import numpy as np
from pathlib import Path

#
#
# STEP 1: 
# Build balanced triplets
#

def build_balanced_triplets(df: pd.DataFrame) -> pd.DataFrame:
    triplets = []

    for row in df.itertuples():
        query_id = row.query_id
        query_text = row.query
        passages = row.passages

        if not isinstance(passages, dict):
            continue

        pos_texts = [txt for sel, txt in zip(passages['is_selected'], passages['passage_text'])
                     if sel == 1 and isinstance(txt, str) and txt.strip()]

        neg_texts_pool = [txt for sel, txt in zip(passages['is_selected'], passages['passage_text'])
                          if sel == 0 and isinstance(txt, str) and txt.strip()]

        if not pos_texts or not neg_texts_pool:
            continue

        # Sample same number of negatives as positives
        neg_texts = random.sample(neg_texts_pool, min(len(pos_texts), len(neg_texts_pool)))

        for pos, neg in zip(pos_texts, neg_texts):
            triplets.append({
                'query': query_text,
                'positive_passage': pos,
                'negative_passage': neg
            })

    return pd.DataFrame(triplets)

#
#
# STEP 2:
# shuffle and split into train/val/test
#

def shuffle_and_split_triplets(triplets_df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1):
    triplets_df = triplets_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(triplets_df)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_df = triplets_df.iloc[:n_train]
    val_df = triplets_df.iloc[n_train:n_train + n_val]
    test_df = triplets_df.iloc[n_train + n_val:]

    return train_df, val_df, test_df

#
#
#
#
#

def main():
    random.seed(42)
    df = pd.read_parquet("../combined.parquet")
    print(f"Loaded {len(df)} queries.")

    triplets_df = build_balanced_triplets(df)
    print(f"Generated {len(triplets_df)} balanced triplets.")

    train_df, val_df, test_df = shuffle_and_split_triplets(triplets_df)
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    train_df.to_parquet("../train.parquet", index=False)
    val_df.to_parquet("../validation.parquet", index=False)
    test_df.to_parquet("../test.parquet", index=False)

    # Sample preview
    print("\nSample triplets:")
    print(train_df.sample(5, random_state=42)[['query', 'positive_passage', 'negative_passage']].to_string(index=False))


if __name__ == "__main__":
    main()