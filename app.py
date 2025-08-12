import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import time
from transformers import pipeline
import re # Importado para extrair o ID do produto

# --- Funções do Backend OTIMIZADAS com Cache ---

@st.cache_resource # Cache para o recurso (modelo de IA)
def carregar_modelo():
    """Carrega o modelo de IA uma única vez."""
    st.write("🧠 Cache miss: Carregando modelo de IA...")
    return pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

@st.cache_data  # Cache para os dados coletados do Reclame Aqui
def coletar_reviews_ra(url, num_paginas=5):
    """Coleta reviews de uma URL base do Reclame Aqui."""
    st.write(f"🔎 Cache miss: Coletando reviews do Reclame Aqui...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://www.google.com/',
    }
    lista_reclamacoes = []
    session = requests.Session()
    session.headers.update(headers)
    for i in range(1, num_paginas + 1):
        url_pagina = f"{url.strip('/')}/lista-reclamacoes/?pagina={i}"
        try:
            response = session.get(url_pagina, timeout=15)
            response.raise_for_status()
            session.headers.update({'Referer': url_pagina})
            soup = BeautifulSoup(response.content, 'html.parser')
            reclamacoes_na_pagina = soup.find_all('div', class_='sc-1a6092-1')
            if not reclamacoes_na_pagina: break
            for reclamacao_html in reclamacoes_na_pagina:
                titulo = reclamacao_html.find('h4', {'data-testid': 'complaint-title'})
                texto = reclamacao_html.find('p', {'data-testid': 'complaint-description'})
                if titulo and texto:
                    lista_reclamacoes.append(titulo.get_text(strip=True) + " - " + texto.get_text(strip=True))
            time.sleep(1)
        except requests.exceptions.HTTPError:
            st.error(f"Bloqueado ao tentar acessar a página {i} do Reclame Aqui. Tente novamente mais tarde.")
            break
        except requests.exceptions.RequestException:
            continue
    return pd.DataFrame(lista_reclamacoes, columns=['comentario'])

def extrair_id_produto_ml(url):
    """Extrai o ID do produto (ex: MLB123456) de uma URL do Mercado Livre."""
    match = re.search(r'(MLB\d+)', url.upper())
    if match:
        return match.group(1)
    return None

@st.cache_data # Cache para os dados coletados do Mercado Livre
def coletar_reviews_ml(id_produto, num_paginas=5):
    """Coleta reviews da API do Mercado Livre usando o ID do produto."""
    st.write(f"🔎 Cache miss: Coletando reviews do Mercado Livre (API)...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Origin': 'https://www.mercadolivre.com.br',
    }
    lista_reviews = []
    session = requests.Session()
    session.headers.update(headers)
    for i in range(num_paginas):
        offset = i * 5
        url_api = f"https://api.mercadolibre.com/reviews/item/{id_produto}"
        try:
            response = session.get(url_api, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data.get('reviews'): break
            for review in data['reviews']:
                if review.get('content'):
                    lista_reviews.append(review['content'])
            time.sleep(1)
        except requests.exceptions.RequestException:
            st.error(f"Falha ao acessar a API do Mercado Livre. O produto pode não ter reviews ou a API está indisponível.")
            break
    return pd.DataFrame(lista_reviews, columns=['comentario'])

def analisar_dados(df, analisador):
    """Aplica o modelo de IA e retorna o DataFrame enriquecido."""
    if df.empty:
        return df
    comentarios = df['comentario'].tolist()
    resultados_sentimento = analisador(comentarios)
    def mapear_sentimento(resultado):
        score = int(resultado['label'].split(' ')[0])
        if score <= 2: return 'negative'
        if score == 3: return 'neutral'
        return 'positive'
    df['sentimento_label'] = [mapear_sentimento(res) for res in resultados_sentimento]
    return df

def exibir_dashboard(df):
    """Desenha os componentes visuais do dashboard."""
    st.markdown("---")
    st.subheader("📊 Resumo dos Sentimentos")
    contagem_sentimentos = df['sentimento_label'].value_counts()
    fig_pizza = px.pie(
        values=contagem_sentimentos.values, 
        names=contagem_sentimentos.index, 
        title='Distribuição de Sentimentos',
        color=contagem_sentimentos.index,
        color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'blue'}
    )
    st.plotly_chart(fig_pizza, use_container_width=True)
    st.subheader("💬 Amostra de Comentários")
    for sentimento in ['positive', 'negative', 'neutral']:
        # Verifica se há comentários daquele sentimento antes de criar o expander
        df_sentimento = df[df['sentimento_label'] == sentimento]
        if not df_sentimento.empty:
            with st.expander(f"Ver comentários da categoria '{sentimento.capitalize()}' ({len(df_sentimento)})"):
                for comentario in df_sentimento['comentario'].head():
                    st.write(f"- {comentario}")

# --- Interface Principal da Aplicação ---

st.set_page_config(layout="wide")
st.title("🤖 Analisador de Sentimentos Multi-site")
st.markdown("Cole o link de uma página de empresa do **Reclame Aqui** ou de um produto do **Mercado Livre**.")

analisador = carregar_modelo()

if 'resultado_df' not in st.session_state:
    st.session_state.resultado_df = None

with st.form("url_form"):
    url_input = st.text_input("URL do site para análise")
    submitted = st.form_submit_button("Analisar!")

if submitted:
    if url_input:
        with st.spinner("Iniciando análise... ☕"):
            try:
                df_coletado = pd.DataFrame()
                
                if "mercadolivre.com.br" in url_input.lower():
                    st.info("Site do Mercado Livre detectado.")
                    id_produto = extrair_id_produto_ml(url_input)
                    if id_produto:
                        df_coletado = coletar_reviews_ml(id_produto)
                    else:
                        st.error("Não foi possível encontrar um ID de produto (Ex: MLB123456) na URL do Mercado Livre.")
                
                elif "reclameaqui.com.br" in url_input.lower():
                    st.info("Site do Reclame Aqui detectado.")
                    df_coletado = coletar_reviews_ra(url_input)
                
                else:
                    st.error("Site não suportado. Por favor, insira uma URL válida do Reclame Aqui ou do Mercado Livre.")

                if not df_coletado.empty:
                    st.info(f"Analisando {len(df_coletado)} reviews coletados...")
                    df_analisado = analisar_dados(df_coletado, analisador)
                    st.session_state.resultado_df = df_analisado
                    st.success("Análise concluída!")
                elif url_input:
                    st.warning("Coleta finalizada, mas nenhum review foi encontrado.")
                    st.session_state.resultado_df = None
            except Exception as e:
                st.error(f"Ocorreu um erro geral no processo: {e}")
                st.session_state.resultado_df = None
    else:
        st.error("Por favor, insira uma URL para análise.")

if st.session_state.resultado_df is not None and not st.session_state.resultado_df.empty:
    exibir_dashboard(st.session_state.resultado_df)