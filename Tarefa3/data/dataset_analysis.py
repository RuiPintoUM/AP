import pandas as pd

global min_words 
global max_words 

min_words = float('inf')  # Inicializa com infinito positivo
max_words = 0

# Função para contar palavras
def count_words(text):
    global min_words, max_words  # Referencia as variáveis globais
    if not isinstance(text, str):  # Verifica se é string
        return 0
    
    nb = len(text.split()) 
    
    if nb == 0 or nb < min_words:
        min_words = nb
        
    if nb > max_words:
        max_words = nb
    
    return nb  # Retorna o número de palavras

# Carregar o dataset (ajuste o caminho e separador conforme necessário)
df = pd.read_csv("dataset3_inputs.csv", sep=";", on_bad_lines="skip")

# Calcular o número de palavras por entrada
df["word_count"] = df["Text"].apply(count_words)

# Calcular a média de palavras no dataset
mean_words = df["word_count"].mean()
print(f"Média de palavras por entrada: {mean_words}")

print(f"Mínimo de palavras {min_words}")

print(f"Máximo de palavras {max_words}")