import pandas as pd
import os
from pathlib import Path

# Define the data directory
data_dir = Path(__file__).parent

# Define file paths for the downloaded parquet files
train_file = data_dir / "train-00000-of-00001.parquet"
validation_file = data_dir / "validation-00000-of-00001.parquet"
test_file = data_dir / "test-00000-of-00001.parquet"

# Load each split using pandas
print(f"Loading train data from {train_file}")
train_data = pd.read_parquet(train_file)

print(f"Loading validation data from {validation_file}")
validation_data = pd.read_parquet(validation_file)

print(f"Loading test data from {test_file}")
test_data = pd.read_parquet(test_file)

# Combine all datasets
print("Combining datasets...")
combined_data = pd.concat([train_data, validation_data, test_data], ignore_index=True)

# Create output directory if it doesn't exist
output_dir = data_dir / "ms_marco_combined"
os.makedirs(output_dir, exist_ok=True)

# Save the combined dataset
output_file = output_dir / "combined.parquet"
combined_data.to_parquet(output_file)

print(f"Combined dataset saved to {output_file}")
print(f"Total records: {len(combined_data):,}")
