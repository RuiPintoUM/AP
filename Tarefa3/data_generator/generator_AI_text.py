import google.generativeai as genai
import csv
import time

GOOGLE_API_KEY = 'AIzaSyDlTl5fVv3_dJEkKlIgnQvBl1vwPpOdyXA'
genai.configure(api_key=GOOGLE_API_KEY)

prompt = "Escreva um texto de aproximadamente 140 palavras que conecte fenômenos naturais e avanços humanos de forma ampla. Inicie explorando um processo ou evento fundamental da natureza, como a formação de algo no universo, na Terra ou em organismos vivos. Depois, relacione isso a um aspecto da existência humana, como saúde, comportamento ou adaptação ao ambiente. Inclua uma perspectiva sobre como a ciência ou a tecnologia ajudam a compreender ou moldar esse cenário, seja por meio de descobertas, inovações ou soluções para desafios. Finalize com uma reflexão sobre o impacto disso no presente ou no futuro, considerando benefícios, riscos ou mistérios ainda não resolvidos. Mantenha o tom acessível e o conteúdo variado, permitindo diferentes interpretações e exemplos a cada uso."

def gerar_resposta(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Erro ao gerar resposta: {e}")
        return None

dados = []
x = 0
dimension = 100
delay = 2  # Aumenta o atraso para 2 segundos

while x < dimension:
    resposta = gerar_resposta(prompt)
    if resposta:
        dados.append([resposta, "AI"])
        time.sleep(delay)  # Espera antes da próxima solicitação
        x += 1
    else:
        break

with open('dataset_respostas_ia.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Resposta", "IA"])
    writer.writerows(dados)

print("O dataset foi gerado e salvo no arquivo 'dataset_respostas_ia.csv'.")