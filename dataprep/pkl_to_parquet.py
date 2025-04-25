import pandas as pd

# Step 1: Load your .pkl file
df = pd.read_pickle("train_tokenised_hard.pkl")

# Step 2: Save as .parquet
df.to_parquet("train_tokenised_hard.parquet", index=False)
