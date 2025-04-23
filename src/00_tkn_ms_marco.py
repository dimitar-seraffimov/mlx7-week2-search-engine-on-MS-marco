import pandas as pd
import pickle
import requests
import tqdm
import re
from collections import Counter
from pathlib import Path
import json

#
# CONFIGURATION
#

COMBINED_PARQUET = Path("../combined.parquet")
TRAIN_PARQUET = Path("../train.parquet")
VALID_PARQUET = Path("../validation.parquet")
TEST_PARQUET = Path("../test.parquet")

#
# TEXT PREPROCESSING
#

def preprocess(text: str) -> list[str]:
    text = text.lower()
    replacements = {
        r"\.": " <PERIOD> ",
        r",":  " <COMMA> ",
        r'"':  " <QUOTATION_MARK> ",
        r";":  " <SEMICOLON> ",
        r"!":  " <EXCLAMATION_MARK> ",
        r"\?": " <QUESTION_MARK> ",
        r"\(": " <LEFT_PAREN> ",
        r"\)": " <RIGHT_PAREN> ",
        r"--": " <HYPHENS> ",
        r":":  " <COLON> ",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text.split()

#
#
#
#
#

def text_to_ids(text: str, vocab_to_int: dict[str, int], max_unk_ratio: float = 0.2) -> list[int]:
    unk_id = vocab_to_int["<UNK>"]
    words = preprocess(text)
    ids = [vocab_to_int.get(word, unk_id) for word in words]
    unk_count = sum(1 for i in ids if i == unk_id)
    if len(ids) == 0 or (unk_count / len(ids)) > max_unk_ratio:
        return []
    return ids

#
#
#
#
#

def display_triplets(df: pd.DataFrame, int_to_vocab: dict[int, str], n: int = 5):
    print("\nSample tokenised triplets:")

    shown = 0
    for _, row in df.iterrows():
        if not row["query_ids"] or not row["pos_ids"] or not row["neg_ids"]:
            continue
        print("\n---")
        print(f"### Query: {row['query']}")
        print(f"   Tokens: {row['query_ids']}")
        print(f"   Words:  {' '.join([int_to_vocab.get(i, '<UNK>') for i in row['query_ids']])}")
        print(f"+++ Positive: {row['positive_passage'][:100]}...")
        print(f"   Tokens: {row['pos_ids'][:10]}... ({len(row['pos_ids'])} tokens)")
        print(f"   Words:  {' '.join([int_to_vocab.get(i, '<UNK>') for i in row['pos_ids'][:10]])}...")
        print(f"--- Negative: {row['negative_passage'][:100]}...")
        print(f"   Tokens: {row['neg_ids'][:10]}... ({len(row['neg_ids'])} tokens)")
        print(f"   Words:  {' '.join([int_to_vocab.get(i, '<UNK>') for i in row['neg_ids'][:10]])}...")
        shown += 1
        if shown == n:
            break
  
#
#
#
#
#

def process_ms_marco() -> tuple[list[str], list[str]]:
    print("Preprocessing MS MARCO queries and passages...")

    df = pd.read_parquet(COMBINED_PARQUET)
    query_tokens = []
    passage_tokens = []
    
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing MS MARCO"):
        query_tokens.extend(preprocess(str(row["query"])))
        for passage in row["passages"]["passage_text"]:
            passage_tokens.extend(preprocess(str(passage)))
    return query_tokens, passage_tokens

#
# 
# Build and save vocabulary
#
#

def build_and_save_vocab(corpus: list[str]):
    print("Building vocabulary...")

    word_counts = Counter(corpus)

    # filter out words that appear less than 5 times
    filtered_corpus = [w for w in corpus if word_counts[w] > 5]
    # sort words by frequency
    sorted_vocab = sorted(set(filtered_corpus), key=lambda w: word_counts[w], reverse=True)

    # create int to vocab mapping
    int_to_vocab = {idx + 1: word for idx, word in enumerate(sorted_vocab)}
    int_to_vocab[0] = "<PAD>"
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    # add <UNK> token to vocab at the end
    unk_id = len(vocab_to_int)
    vocab_to_int["<UNK>"] = unk_id
    int_to_vocab[unk_id] = "<UNK>"

    # Save to root directory
    pickle.dump(vocab_to_int, open("../tkn_vocab_to_int.pkl", "wb"))
    pickle.dump(int_to_vocab, open("../tkn_int_to_vocab.pkl", "wb"))

    print(f"Vocabulary saved with {len(vocab_to_int):,} tokens (including <PAD>, <UNK>)")
    print(f"Vocab size (excluding <PAD>, <UNK>): {len(vocab_to_int) - 2:,}")
    return vocab_to_int, int_to_vocab

#
#
# Tokenise (queries and positive/negative passages) and save splits (train, validation, test)
#
#

def extract_passages(passages):
    # if passages is a JSON string, decode it
    if isinstance(passages, str):
        try:
            passages = json.loads(passages)
        except json.JSONDecodeError:
            return pd.Series(["", ""])  # fallback if broken

    if not isinstance(passages, list):
        return pd.Series(["", ""])

    pos = next((p.get("passage_text", "") for p in passages if isinstance(p, dict) and p.get("is_selected", False)), "")
    neg = next((p.get("passage_text", "") for p in passages if isinstance(p, dict) and not p.get("is_selected", False)), "")
    return pd.Series([pos, neg])

def tokenise_and_save_splits(vocab_to_int: dict, int_to_vocab: dict):
    print("Tokenising data splits...")

    splits = {
        "train": pd.read_parquet(TRAIN_PARQUET),
        "validation": pd.read_parquet(VALID_PARQUET),
        "test": pd.read_parquet(TEST_PARQUET)
    }

    for name, df in splits.items():
        # extract pos/neg from 'passages' field
        df[["positive_passage", "negative_passage"]] = df["passages"].apply(extract_passages)

        df["query_ids"] = df["query"].apply(lambda x: text_to_ids(x, vocab_to_int))
        df["pos_ids"]   = df["positive_passage"].apply(lambda x: text_to_ids(x, vocab_to_int))
        df["neg_ids"]   = df["negative_passage"].apply(lambda x: text_to_ids(x, vocab_to_int))

        df["q_len"] = df["query_ids"].str.len()
        df["p_len"] = df["pos_ids"].str.len()
        df["n_len"] = df["neg_ids"].str.len()

        print(f"{name}: avg q {df['q_len'].mean():.1f}, p {df['p_len'].mean():.1f}, n {df['n_len'].mean():.1f}")
        df.to_pickle(f"../{name}_tokenised.pkl")

        print(f"\nTokenised samples from {name.upper()}:")
        display_triplets(df, int_to_vocab, n=3)

#
#
#
#
#

if __name__ == "__main__":
    query_tokens, passage_tokens = process_ms_marco()

    # Save to root directory
    pickle.dump(query_tokens, open("../ms_marco_queries.pkl", "wb"))
    pickle.dump(passage_tokens, open("../ms_marco_passages.pkl", "wb"))

    combined_corpus = query_tokens + passage_tokens
    pickle.dump(combined_corpus, open("../combined_corpus.pkl", "wb"))
    print(f"Combined corpus has {len(combined_corpus):,} tokens")

    vocab_to_int, int_to_vocab = build_and_save_vocab(combined_corpus)
    tokenise_and_save_splits(vocab_to_int, int_to_vocab)