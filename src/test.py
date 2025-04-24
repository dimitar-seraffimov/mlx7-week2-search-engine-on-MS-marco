import pandas as pd
from pathlib import Path
from tqdm import tqdm

CHUNKS = 16  # Adjust based on memory and performance
COMBINED_PARQUET = Path("../combined.parquet")
VOCAB_PATH = Path("../tkn_vocab_to_int.parquet")
OUTPUT_DIR = Path("../tokenised_chunks")
OUTPUT_DIR.mkdir(exist_ok=True)

from s02_tkn_ms_marco import text_to_ids

def tokenise_and_save_full_corpus_chunked(vocab_to_int):
    print("\nTokenising full passage corpus (chunked mode)...")

    df = pd.read_parquet(COMBINED_PARQUET)
    if "passages" not in df.columns:
        raise KeyError("'passages' column not found.")

    df = df.rename(columns={"passages": "passage"})
    df = df.drop_duplicates(subset=["passage"]).reset_index(drop=True)
    df["id"] = df.index.map(lambda i: f"doc_{i}")

    chunk_size = len(df) // CHUNKS + 1
    chunks = [df.iloc[i:i+chunk_size].copy() for i in range(0, len(df), chunk_size)]

    for i, chunk in enumerate(chunks):
        chunk_file = OUTPUT_DIR / f"combined_tokenised_part_{i}.parquet"
        if chunk_file.exists():
            print(f"[✓] Skipping already-processed chunk {i}")
            continue

        print(f"[→] Processing chunk {i}/{len(chunks)}...")
        tqdm.pandas(desc=f"Tokenising chunk {i}")
        chunk["pos_ids"] = chunk["passage"].progress_apply(lambda x: text_to_ids(x, vocab_to_int))
        chunk["p_len"] = chunk["pos_ids"].str.len()
        chunk[["id", "passage", "pos_ids", "p_len"]].to_parquet(chunk_file, index=False)
        print(f"[✓] Saved: {chunk_file}")

    print("\nAll chunks processed. To merge them into final file:")
    print("pd.concat([pd.read_parquet(f) for f in Path('../tokenised_chunks').glob('combined_tokenised_part_*.parquet')])\n" 
          ".to_parquet('../combined_tokenised.parquet', index=False)")

if __name__ == "__main__":
    # Load vocab
    vocab_to_int = pd.read_parquet(VOCAB_PATH).iloc[0].to_dict()

    # Run chunked tokenisation
    tokenise_and_save_full_corpus_chunked(vocab_to_int)