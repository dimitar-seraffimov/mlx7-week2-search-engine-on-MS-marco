import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm

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
    # collect all passage texts once
    all_passages = [
        txt
        for row in df.itertuples()
        if isinstance(row.passages, dict)
        for txt in row.passages["passage_text"]
        if isinstance(txt, str) and txt.strip()
    ]

    print(f"[Info] Total unique passages for global sampling: {len(all_passages):,}")

    triplets = []
    all_passages_clean = list(set(all_passages))

    for row in tqdm(df.itertuples(), total=len(df), desc="Building triplets"):
        if not isinstance(row.passages, dict):
            continue

        query = row.query
        pos_passages = [
            txt for txt in row.passages["passage_text"]
            if isinstance(txt, str) and txt.strip()
        ]
        if not pos_passages:
            continue

        # sample negatives randomly from the global pool
        sampled_negatives = random.sample(all_passages_clean, len(pos_passages))

        for pos, neg in zip(pos_passages, sampled_negatives):
            triplets.append({
                "query": query,
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

    #print("\n[Sample Triplets]")
    #print(train_df.sample(5)[["query", "positive_passage", "negative_passage"]].to_string(index=False))
    
    query_counts = triplets_df.groupby("query").size()
    print("\nTriplet count per query (first 10):")
    print(query_counts.head(10))
    print(f"\nAverage triplets per query: {query_counts.mean():.2f}")

if __name__ == "__main__":
    main()
