import pandas as pd
import pickle
import requests
import tqdm
import re
from collections import Counter
from pathlib import Path
from tqdm import tqdm

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
# TOKENISE TEXT
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
# DISPLAY SAMPLE TOKENISED TRIPLETS
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
# EXTRACT TOKENS FROM TRIPLETS
#
#

def extract_tokens_from_triplets(parquet_files: list[Path]) -> list[str]:
    print("Extracting tokens from triplets...")

    tokens = []
    for file in parquet_files:
        print(f"\n→ Processing: {file.name}")
        df = pd.read_parquet(file)

        for col in ['query', 'positive_passage', 'negative_passage']:
            print(f"   - Tokenizing column: {col}")
            for text in tqdm(df[col], desc=f"   Processing {col}", leave=False):
                tokens.extend(preprocess(str(text)))

    return tokens
#
#
# BUILD & SAVE VOCABULARY
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
    pd.DataFrame(vocab_to_int, index=[0]).to_parquet("../tkn_vocab_to_int.parquet")
    pd.DataFrame(int_to_vocab, index=[0]).to_parquet("../tkn_int_to_vocab.parquet")

    print(f"Vocabulary saved with {len(vocab_to_int):,} tokens (including <PAD>, <UNK>)")
    print(f"Vocab size (excluding <PAD>, <UNK>): {len(vocab_to_int) - 2:,}")
    return vocab_to_int, int_to_vocab

#
#
# TOKENISE & SAVE SPLITS
#
#

def tokenise_and_save_splits(vocab_to_int: dict, int_to_vocab: dict):
    print("Tokenising data splits...")

    splits = {
        "train": pd.read_parquet(TRAIN_PARQUET),
        "validation": pd.read_parquet(VALID_PARQUET),
        "test": pd.read_parquet(TEST_PARQUET)
    }

    for name, df in splits.items():
        df["query_ids"] = df["query"].apply(lambda x: text_to_ids(x, vocab_to_int))
        df["pos_ids"]   = df["positive_passage"].apply(lambda x: text_to_ids(x, vocab_to_int))
        df["neg_ids"]   = df["negative_passage"].apply(lambda x: text_to_ids(x, vocab_to_int))

        df["q_len"] = df["query_ids"].str.len()
        df["p_len"] = df["pos_ids"].str.len()
        df["n_len"] = df["neg_ids"].str.len()

        print(f"{name}: avg q {df['q_len'].mean():.1f}, p {df['p_len'].mean():.1f}, n {df['n_len'].mean():.1f}")
        # Save to root directory
        df.to_parquet(f"../{name}_tokenised.parquet")

        print(f"\nTokenised samples from {name.upper()}:")
        display_triplets(df, int_to_vocab, n=3)

#
#
#
#
#

if __name__ == "__main__":

    # build vocab from already-created triplets
    parquet_files = [TRAIN_PARQUET, VALID_PARQUET, TEST_PARQUET]

    # save combined corpus
    combined_corpus = extract_tokens_from_triplets(parquet_files)
    pd.DataFrame({"token": combined_corpus}).to_parquet("../combined_corpus.parquet", index=False)
    print(f"Combined corpus has {len(combined_corpus):,} tokens")

    # build vocab
    vocab_to_int, int_to_vocab = build_and_save_vocab(combined_corpus)
    # tokenise and save splits
    tokenise_and_save_splits(vocab_to_int, int_to_vocab)