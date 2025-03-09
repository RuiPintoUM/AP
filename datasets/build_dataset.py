import pandas as pd

# Carregar os datasets
inputs_df = pd.read_csv('dataset1_inputs.csv', sep='\t')
outputs_df = pd.read_csv('dataset1_outputs.csv', sep='\t')

# Combinar os datasets com base no ID
combined_df = pd.merge(inputs_df, outputs_df, on='ID')

# Salvar o novo dataset combinado
combined_df.to_csv('combined_dataset.csv', index=False)