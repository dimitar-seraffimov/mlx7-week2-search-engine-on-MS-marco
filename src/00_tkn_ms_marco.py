# /src/00_tkn_ms_marco.py

import pandas as pd
import numpy as np
import random
import pickle
import tqdm
import re
from collections import Counter
from pathlib import Path

COMBINED_PARQUET = Path("../combined.parquet")

TRAIN_PICKLE = Path("../train_tokenised.pkl")
VALID_PICKLE = Path("../validation_tokenised.pkl")
TEST_PICKLE  = Path("../test_tokenised.pkl")

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

def text_to_ids(text: str, vocab_to_int: dict[str, int], max_unk_ratio: float = 0.2) -> list[int]:
    unk_id = vocab_to_int["<UNK>"]
    words = preprocess(text)
    ids = [vocab_to_int.get(word, unk_id) for word in words]
    unk_count = sum(1 for i in ids if i == unk_id)
    if len(ids) == 0 or (unk_count / len(ids)) > max_unk_ratio:
        return []
    return ids

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

def extract_triplets(raw_df: pd.DataFrame, offset: int = 4) -> pd.DataFrame:
    triplets = []
    num_rows = len(raw_df)
    for i in range(num_rows):
        row = raw_df.iloc[i]
        query = row['query']
        passages = row['passages']
        if not isinstance(passages, dict):
            continue
        is_selected = passages.get('is_selected', [])
        passage_texts = passages.get('passage_text', [])
        if len(is_selected) == 0 or len(passage_texts) == 0:
            continue
        selected_indices = np.where(np.array(is_selected) == 1)[0]
        if selected_indices.size == 0:
            continue
        positive_passage = passage_texts[selected_indices[0]]
        negative_index = (i + offset) % num_rows
        if negative_index == i:
            continue
        negative_row = raw_df.iloc[negative_index]
        neg_passages = negative_row['passages'].get('passage_text', [])
        if not neg_passages:
            continue
        negative_passage = None
        for _ in range(3):
            candidate = random.choice(neg_passages)
            if candidate != positive_passage:
                negative_passage = candidate
                break
        if not negative_passage:
            continue
        triplets.append({
            'query': query,
            'positive_passage': positive_passage,
            'negative_passage': negative_passage
        })
    return pd.DataFrame(triplets)

def build_and_save_vocab(corpus: list[str]):
    print("Building vocabulary...")
    word_counts = Counter(corpus)
    filtered = [w for w in corpus if word_counts[w] > 5]
    sorted_vocab = sorted(set(filtered), key=lambda w: word_counts[w], reverse=True)
    int_to_vocab = {i + 1: word for i, word in enumerate(sorted_vocab)}
    int_to_vocab[0] = "<PAD>"
    vocab_to_int = {word: i for i, word in int_to_vocab.items()}
    unk_id = len(vocab_to_int)
    vocab_to_int["<UNK>"] = unk_id
    int_to_vocab[unk_id] = "<UNK>"
    pickle.dump(vocab_to_int, open("../tkn_vocab_to_int.pkl", "wb"))
    pickle.dump(int_to_vocab, open("../tkn_int_to_vocab.pkl", "wb"))
    print(f"Vocabulary saved with {len(vocab_to_int):,} tokens (including <PAD>, <UNK>)")
    return vocab_to_int, int_to_vocab

def tokenise_and_save(df: pd.DataFrame, name: str, vocab_to_int: dict, int_to_vocab: dict):
    df["query_ids"] = df["query"].apply(lambda x: text_to_ids(x, vocab_to_int))
    df["pos_ids"] = df["positive_passage"].apply(lambda x: text_to_ids(x, vocab_to_int))
    df["neg_ids"] = df["negative_passage"].apply(lambda x: text_to_ids(x, vocab_to_int))
    df["q_len"] = df["query_ids"].str.len()
    df["p_len"] = df["pos_ids"].str.len()
    df["n_len"] = df["neg_ids"].str.len()
    print(f"{name}: avg q {df['q_len'].mean():.1f}, p {df['p_len'].mean():.1f}, n {df['n_len'].mean():.1f}")
    df.to_pickle(f"../{name}_tokenised.pkl")
    display_triplets(df, int_to_vocab)

if __name__ == "__main__":
    random.seed(42)
    df = pd.read_parquet(COMBINED_PARQUET)
    triplets_df = extract_triplets(df, offset=4)

    combined = triplets_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(combined)
    train_df = combined.iloc[:int(0.8 * n)]
    valid_df = combined.iloc[int(0.8 * n):int(0.9 * n)]
    test_df  = combined.iloc[int(0.9 * n):]

    print(f"Splits → train: {len(train_df)}, val: {len(valid_df)}, test: {len(test_df)}")

    # Build vocab
    all_tokens = []
    for col in ["query", "positive_passage", "negative_passage"]:
        all_tokens.extend([t for row in triplets_df[col] for t in preprocess(row)])
    vocab_to_int, int_to_vocab = build_and_save_vocab(all_tokens)

    # Tokenise each split
    tokenise_and_save(train_df, "train", vocab_to_int, int_to_vocab)
    tokenise_and_save(valid_df, "validation", vocab_to_int, int_to_vocab)
    tokenise_and_save(test_df,  "test", vocab_to_int, int_to_vocab)
