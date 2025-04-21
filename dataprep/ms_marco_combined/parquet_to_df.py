import pandas as pd

# Load Parquet file into a DataFrame
df = pd.read_parquet("combined.parquet")

# View the first few rows
#print(df.head("50"))

print(df[["passages", "query"]])

# Create a new DataFrame with the passages and query
passages_df = pd.DataFrame(df[["passages", "query"]])

# Save the passages DataFrame to a CSV file
passages_df.to_csv("passages_and_query.csv", index=False)
