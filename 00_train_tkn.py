import pickle
import pandas as pd
import re

with open('dataprep/ms_marco_combined/vocab_tkn/tkn_words_to_ids.pkl', 'rb') as f:
    vocab_to_int = pickle.load(f)
with open('dataprep/ms_marco_combined/vocab_tkn/tkn_ids_to_words.pkl', 'rb') as f:
    int_to_vocab = pickle.load(f)

# get special token IDs
PAD_ID = vocab_to_int.get('<PAD>', 0)
UNK_ID = vocab_to_int.get('<UNK>', 1)


#
#
#
#
#

train_df = pd.read_parquet("dataprep/ms_marco_combined/train.parquet")
val_df   = pd.read_parquet("dataprep/ms_marco_combined/validation.parquet")
test_df  = pd.read_parquet("dataprep/ms_marco_combined/test.parquet")
print(f"Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

#
#
#
#
#

def preprocess(text: str) -> list[str]:
    text = text.lower()
    subs = {
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
    for p, tok in subs.items():
        text = re.sub(p, tok, text)
    return text.split()

#
#
#
#
#

def text_to_ids(text: str) -> list[int]:
    return [
        vocab_to_int.get(w, UNK_ID)
        for w in preprocess(text)
        if w  # skip empty tokens
    ]

#
#
#
#
#

for df, name in [(train_df, "train"), (val_df, "validation"), (test_df, "test")]:
    df["query_ids"] = df["query"].apply(text_to_ids)
    df["pos_ids"]   = df["positive_passage"].apply(text_to_ids)
    df["neg_ids"]   = df["negative_passage"].apply(text_to_ids)
    print(f"{name}: avg q {df['query_ids'].str.len().mean():.1f}, "
          f"p {df['pos_ids'].str.len().mean():.1f}, "
          f"n {df['neg_ids'].str.len().mean():.1f}")

    
#
#
#
#
#


train_df.to_pickle("dataprep/ms_marco_combined/tokenised/train_tokenised.pkl")
val_df.to_pickle(  "dataprep/ms_marco_combined/tokenised/validation_tokenised.pkl")
test_df.to_pickle( "dataprep/ms_marco_combined/tokenised/test_tokenised.pkl")
