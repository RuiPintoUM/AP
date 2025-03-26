import wikipedia
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import re

# Baixar os recursos necessários do NLTK
nltk.download('punkt')
nltk.download('punkt_tab')

# Função para verificar se o texto contém equações ou símbolos matemáticos
def has_math_symbols(text):
    math_patterns = [
        r'=', r'\+', r'-', r'\*', r'/', r'\^', r'_',  # Símbolos básicos
        r'\\(?:[a-zA-Z]+|\{.*?\})',  # Qualquer comando LaTeX como \sum, \phi, \frac
        r'\d+\s*[+\-*/^]\s*\d+',  # Expressões como "2 + 3"
        r'[∑∫∂∞π√±×÷]',  # Símbolos matemáticos específicos
    ]
    return any(re.search(pattern, text) for pattern in math_patterns)

# Função para limpar texto, removendo "{", "}", "==", "===" e "References"
def clean_text(text):
    references_index = text.find("References")
    if references_index != -1:
        text = text[:references_index]
    
    text = re.sub(r'\{', '', text)  # Remove "{"
    text = re.sub(r'\}', '', text)  # Remove "}"
    text = re.sub(r'==', '', text)  # Remove "=="
    text = re.sub(r'===', '', text)  # Remove "==="
    text = re.sub(r'\[.*?\]', '', text)  # Remove referências [n]
    text = re.sub(r'\n+', ' ', text)  # Substitui quebras de linha por espaço
    text = re.sub(r'\s+', ' ', text).strip()  # Normaliza espaços
    return text

# Função para dividir texto em trechos de 80-200 palavras
def split_text_into_chunks(text, min_words=80, max_words=200):
    text = clean_text(text)
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        word_count = len(words)
        
        if current_word_count + word_count > max_words:
            if current_word_count >= min_words:
                chunk_text = ' '.join(current_chunk)
                if not has_math_symbols(chunk_text):
                    chunks.append(chunk_text)
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count
    
    if min_words <= current_word_count <= max_words:
        chunk_text = ' '.join(current_chunk)
        if not has_math_symbols(chunk_text):
            chunks.append(chunk_text)
    
    return chunks

# Função para gerar tópicos dinamicamente usando wikipedia.search()
def generate_topics(seed_keywords, results_per_keyword=50):
    wikipedia.set_lang("en")
    topics = set()  # Usar um set para evitar duplicatas
    
    for keyword in seed_keywords:
        try:
            # Buscar tópicos relacionados ao keyword
            search_results = wikipedia.search(keyword, results=results_per_keyword)
            for result in search_results:
                # Filtrar resultados muito curtos ou irrelevantes
                if len(result) > 3 and "disambiguation" not in result.lower():
                    topics.add(result)
        except Exception as e:
            print(f"Erro ao buscar tópicos para {keyword}: {e}")
    
    return list(topics)

# Configurar a biblioteca wikipedia
wikipedia.set_lang("en")

# Seed keywords para gerar tópicos em várias áreas científicas
seed_keywords = [
    "Physics", "Quantum mechanics", "Astrophysics", "Thermodynamics", "Nuclear physics",
    "Chemistry", "Organic chemistry", "Biochemistry", "Chemical engineering", "Periodic table",
    "Geology", "Plate tectonics", "Volcanology", "Paleontology", "Geophysics",
    "Biology", "Genetics", "Ecology", "Microbiology", "Evolution",
    "Medicine", "Immunology", "Epidemiology", "Pharmacology", "Biomedicine",
    "Meteorology", "Climate science", "Weather forecasting", "Atmospheric science",
    "Technology", "Robotics", "Artificial intelligence", "Computer science", "Engineering",
    "Astronomy", "Solar system", "Exoplanets", "Galaxies", "Stellar evolution",
    "History of science", "Philosophy of science", "Science education", "Scientific method"
]

# Gerar tópicos dinamicamente
topics = generate_topics(seed_keywords, results_per_keyword=80)
print(f"Gerados {len(topics)} tópicos únicos.")

# Lista para armazenar os dados
data = []

# Extrair textos da Wikipedia
target_rows = 15000  # Metade do objetivo total
for topic in topics:
    if len(data) >= target_rows:
        break
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        content = page.content
        chunks = split_text_into_chunks(content)
        
        for i, chunk in enumerate(chunks):
            if len(data) >= target_rows:
                break
            data.append({
                "ID": f"W-{len(data) + 1}",
                "Text": chunk,
                "Label": "Human"
            })
            print(f"Extraído: {topic} - Trecho {i+1} ({len(word_tokenize(chunk))} palavras)")
        
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Erro de desambiguação para {topic}: {e.options}")
    except wikipedia.exceptions.PageError:
        print(f"Página não encontrada para {topic}")
    except Exception as e:
        print(f"Erro ao processar {topic}: {e}")

# Criar um DataFrame com os dados
df = pd.DataFrame(data)

# Salvar em CSV
df.to_csv("wikipedia_human_texts.csv", sep=";", index=False)
print(f"Dados salvos em 'wikipedia_human_texts.csv' com {len(df)} linhas")

# Visualizar os primeiros 5 registos
print(df.head())