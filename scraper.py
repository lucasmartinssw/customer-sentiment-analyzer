import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL da página de avaliações 
URL = "https://www.reclameaqui.com.br/empresa/energisa-minas-rio/lista-reclamacoes/"

# O requests pode precisar de um cabeçalho para simular um navegador
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

response = requests.get(URL, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

lista_reviews = []


for review_html in soup.find_all('div', class_='ui-review-capability__summary__plain_text'):
    try:
        texto_review = review_html.find('p', class_='ui-review-capability__summary__plain_text__summary_container').get_text(strip=True)
        lista_reviews.append(texto_review)
    except AttributeError:
        continue

# Salvar em um DataFrame do Pandas e depois em CSV
df = pd.DataFrame(lista_reviews, columns=['comentario'])
df.to_csv('reviews_coletados.csv', index=False)

print(f"Coleta finalizada! {len(lista_reviews)} reviews salvos em 'reviews_coletados.csv'.")