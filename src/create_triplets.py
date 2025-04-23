import pandas as pd
import random
from pathlib import Path

#
#
# SETUP
#
#

DATA_PATH = Path("../combined.parquet")
OUTPUT_TRAIN = Path("../train.parquet")
OUTPUT_VAL = Path("../validation.parquet")
OUTPUT_TEST = Path("../test.parquet")


#
#
# BUILD TRIPLETS
#
#

def build_random_neg_sampled_triplets(df: pd.DataFrame) -> pd.DataFrame:
    all_passages = []

    for row in df.itertuples():
        passages = row.passages
        if not isinstance(passages, dict):
            continue
        all_passages.extend([txt for txt in passages["passage_text"] if isinstance(txt, str) and txt.strip()])

    print(f"[Info] Total unique passages for global negative pool: {len(all_passages):,}")

    triplets = []

    for row in df.itertuples():
        query_text = row.query
        passages = row.passages

        if not isinstance(passages, dict):
            continue

        # all passages linked to this query are positives
        positive_passages = [txt for txt in passages["passage_text"] if isinstance(txt, str) and txt.strip()]
        if not positive_passages:
            continue

        # exclude current positives from negatives
        global_neg_pool = list(set(all_passages) - set(positive_passages))

        # sample same number of negatives as positives
        if len(global_neg_pool) < len(positive_passages):
            continue  # skip edge case

        sampled_negatives = random.sample(global_neg_pool, len(positive_passages))

        for pos, neg in zip(positive_passages, sampled_negatives):
            triplets.append({
                "query": query_text,
                "positive_passage": pos,
                "negative_passage": neg
            })

    return pd.DataFrame(triplets)

#
#
# SPLIT DATA
#
#

def shuffle_and_split_triplets(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return df[:n_train], df[n_train:n_train + n_val], df[n_train + n_val:]

#
#
# RUN
#
#

def main():
    random.seed(42)

    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} queries")

    triplets_df = build_random_neg_sampled_triplets(df)
    print(f"[✓] Created {len(triplets_df):,} triplets")

    train_df, val_df, test_df = shuffle_and_split_triplets(triplets_df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_df.to_parquet(OUTPUT_TRAIN, index=False)
    val_df.to_parquet(OUTPUT_VAL, index=False)
    test_df.to_parquet(OUTPUT_TEST, index=False)

    print("\n[Sample Triplets]")
    print(train_df.sample(5)[["query", "positive_passage", "negative_passage"]].to_string(index=False))

if __name__ == "__main__":
    main()
