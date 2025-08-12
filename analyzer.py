import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# 1. Carregar os dados do seu arquivo CSV
try:
    df = pd.read_csv('reviews_coletados.csv')
    df.dropna(subset=['comentario'], inplace=True)
    comentarios = df['comentario'].tolist()
except FileNotFoundError:
    print("Erro: Arquivo 'reviews_coletados.csv' não encontrado. Certifique-se que ele está na mesma pasta.")
    exit()

# 2. Carregar o pipeline de análise de sentimento com o modelo CORRETO e ESTÁVEL
print("Carregando o modelo de análise de sentimento... (isso pode levar um tempo na primeira vez)")
try:
    # Este é um modelo muito popular e estável
    analisador_sentimento = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Não foi possível carregar o modelo. Erro: {e}")
    exit()

# 3. Analisar os sentimentos
resultados = analisador_sentimento(comentarios)

# 4. TRADUZIR a saída de "estrelas" para "positivo", "negativo" ou "neutro"
# Esta é a nova lógica de mapeamento
def mapear_sentimento(resultado):
    score = int(resultado['label'].split(' ')[0]) # Pega o número da estrela (ex: "5 stars" -> 5)
    if score <= 2:
        return 'negative'
    elif score == 3:
        return 'neutral'
    else: # 4 ou 5 estrelas
        return 'positive'

df['sentimento_label'] = [mapear_sentimento(res) for res in resultados]
df['sentimento_score'] = [res['score'] for res in resultados]


# 5. Extração de Tópicos (Topic Modeling) - Sem alterações aqui
print("Extraindo tópicos dos comentários...")
comentarios_positivos = df[df['sentimento_label'] == 'positive']['comentario']
comentarios_negativos = df[df['sentimento_label'] == 'negative']['comentario']

def extrair_topicos(comentarios, n_topicos=3, n_palavras=5):
    if len(comentarios) < n_topicos:
        return {}
    
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'meu', 'minha'])
    tfidf = vectorizer.fit_transform(comentarios)
    
    nmf = NMF(n_components=n_topicos, random_state=1, l1_ratio=.5).fit(tfidf)
    
    feature_names = vectorizer.get_feature_names_out()
    topicos = {}
    for i, topic in enumerate(nmf.components_):
        topicos[f"Tópico {i+1}"] = [feature_names[j] for j in topic.argsort()[:-n_palavras - 1:-1]]
    return topicos

topicos_positivos = extrair_topicos(comentarios_positivos)
topicos_negativos = extrair_topicos(comentarios_negativos)

# 6. Salvar e mostrar os resultados
df.to_csv('reviews_analisados.csv', index=False)

print("\nAnálise concluída com sucesso!")
print("\n--- Tópicos Positivos Identificados ---")
print(topicos_positivos)
print("\n--- Tópicos Negativos Identificados ---")
print(topicos_negativos)
print("\nArquivo 'reviews_analisados.csv' foi gerado com os resultados.")