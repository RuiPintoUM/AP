import pandas as pd
import glob



# Define the path to the parquet files
parquet_files_path = 'train-*.parquet'

# Find all parquet files in the directory
parquet_files = glob.glob(parquet_files_path)

# Initialize an empty list to hold dataframes
dfs = []

# Loop through all parquet files and read them into dataframes
for file in parquet_files:
    df = pd.read_parquet(file, engine='pyarrow')
    
    # Transform the data to the required format
    df['ID'] = df['id'].apply(lambda x: f"D1-{x+1}")  # Create the ID in the required format
    df['Text'] = df['text']
    df['Label'] = df['source'].apply(lambda x: 'Human' if x.lower() == 'human' else 'AI')
    
    # Select and reorder columns
    df = df[['ID', 'Text', 'Label']]
    
    # Append the dataframe to the list
    dfs.append(df)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataframe to a CSV file
csv_file = 'combined_dataset.csv'
combined_df.to_csv(csv_file, index=False, quoting=1)

print(f"All parquet files have been merged and saved to {csv_file}")
