import google.generativeai as genai
import csv
import time

GOOGLE_API_KEY = 'AIzaSyDlTl5fVv3_dJEkKlIgnQvBl1vwPpOdyXA'
genai.configure(api_key=GOOGLE_API_KEY)

prompts = [
    "Write a concise, informative paragraph (between 100 and 180 words) on a specified scientific or technical topic. The text should explore key concepts, offering specific details, examples, or implications, while maintaining a clear and logical flow. Adopt a formal yet approachable tone for an educated audience, using relevant terminology or principles tied to the topic. Vary the approach—such as focusing on historical context, practical applications, theoretical foundations, or current research—while avoiding excessive jargon or speculation beyond established ideas. Ensure the paragraph is self-contained, providing unique insight into the subject without needing external context. Adapt the structure and perspective (e.g., explanatory, analytical, or narrative) to create distinct outputs each time. Topic: [Insert topic here].",
]

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
dimension = 1
delay = 1  # Aumenta o atraso para 2 segundos

while x < dimension:
    for prompt in prompts:
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