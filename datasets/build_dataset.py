import pandas as pd

# ------------ Parte de teste ------------  


# Carregar os datasets
inputs_df = pd.read_csv('dataset1_inputs.csv', sep='\t')
outputs_df = pd.read_csv('dataset1_outputs.csv', sep='\t')

# Combinar os datasets com base no ID
combined_df = pd.merge(inputs_df, outputs_df, on='ID')

# Salvar o novo dataset combinado
combined_df.to_csv('combined_dataset_test.csv', index=False)



# ------------ Parte de treino ------------


# Define o caminho do arquivo CSV
csv_file_path = 'data_set.csv'

# Lê o arquivo CSV em um dataframe
df = pd.read_csv(csv_file_path)

# Criar uma coluna 'ID' usando o índice da linha (já que não há uma coluna 'id')
df['ID'] = df.index.map(lambda x: f"D1-{x+1}")  # Criando IDs sequenciais como D1-1, D1-2, ...

# Renomear colunas conforme necessário
df.rename(columns={'abstract': 'Text'}, inplace=True)

# Criar a coluna 'Label' com base na coluna 'ai_generated'
df['Label'] = df['ai_generated'].apply(lambda x: 'AI' if x else 'Human')

# Selecionar e reordenar colunas
df = df[['ID', 'Text', 'Label']]

# Salvar o novo dataset transformado
output_csv_file = 'combined_dataset_treino.csv'
df.to_csv(output_csv_file, index=False, quoting=1)

print(f"O arquivo CSV foi transformado e salvo como {output_csv_file}")