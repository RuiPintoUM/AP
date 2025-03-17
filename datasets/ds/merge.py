import pandas as pd

def balance_and_merge_datasets(file1, file2, output_file):
    # Load the datasets
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Ensure both datasets have the same column names
    df1.columns = ['Text', 'Label'] if len(df1.columns) == 2 else df1.columns
    df2.columns = ['Text', 'Label'] if len(df2.columns) == 2 else df2.columns
    
    # Merge datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Check if 'Label' column exists
    if 'Label' not in combined_df.columns:
        raise ValueError("The dataset must contain a 'Label' column.")
    
    # Determine the minimum count of entries per label
    min_count = combined_df['Label'].value_counts().min()
    
    # Balance the dataset by sampling an equal number of entries per label
    balanced_df = combined_df.groupby('Label').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
    
    # Save the balanced dataset to a new CSV file
    balanced_df.to_csv(output_file, index=False)
    
    print(f"Balanced dataset saved to {output_file}")

# Example usage
balance_and_merge_datasets('LLM.csv', 'dataset_praquet.csv', 'balanced_output.csv')
