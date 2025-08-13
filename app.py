import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import time
from transformers import pipeline
import re

# --- Funções do Backend OTIMIZADAS com Cache ---

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo de IA uma única vez."""
    st.write("🧠 Cache miss: Carregando modelo de IA...")
    return pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

@st.cache_data(ttl=3600)
def coletar_reviews_ra(url, num_paginas=5):
    """Coleta reviews de uma URL base do Reclame Aqui."""
    st.write(f"🔎 Cache miss: Coletando reviews do Reclame Aqui...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    }
    lista_reclamacoes = []
    session = requests.Session()
    session.headers.update(headers)
    for i in range(1, num_paginas + 1):
        url_pagina = f"{url.strip('/')}/lista-reclamacoes/?pagina={i}"
        try:
            response = session.get(url_pagina, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            reclamacoes_na_pagina = soup.find_all('div', class_='sc-1a6092-1')
            if not reclamacoes_na_pagina: break
            for reclamacao_html in reclamacoes_na_pagina:
                titulo = reclamacao_html.find('h4', {'data-testid': 'complaint-title'})
                texto = reclamacao_html.find('p', {'data-testid': 'complaint-description'})
                if titulo and texto:
                    lista_reclamacoes.append(titulo.get_text(strip=True) + " - " + texto.get_text(strip=True))
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao acessar Reclame Aqui (página {i}): {e}")
            break
    return pd.DataFrame(lista_reclamacoes, columns=['comentario'])

def extrair_id_produto_ml(url):
    """Extrai o ID do produto (ex: MLB123456) de uma URL do Mercado Livre."""
    match = re.search(r'(MLB\d+)', url.upper())
    if match:
        return match.group(1)
    return None

@st.cache_data(ttl=3600)
def coletar_reviews_ml(id_produto, access_token):
    """Coleta reviews da API do Mercado Livre."""
    st.write(f"🔎 Cache miss: Iniciando coleta de reviews do Mercado Livre (API)...")
    headers = {'Authorization': f'Bearer {access_token}'}
    all_reviews_data = []
    session = requests.Session()
    session.headers.update(headers)
    offset = 0
    limit = 50
    while True:
        url_api = f"https://api.mercadolibre.com/reviews/item/{id_produto}"
        try:
            response = session.get(url_api, timeout=15)
            response.raise_for_status()
            data = response.json()
            reviews = data.get('reviews', [])
            if not reviews:
                break
            for review in reviews:
                comentario = review.get('content', '')
                if not comentario:
                    comentario = 'Este review não possui texto.'
                all_reviews_data.append({
                    'comentario': comentario,
                    'nota': review.get('rate', 0)
                })
            total_reviews = data.get('paging', {}).get('total', 0)
            st.info(f"Coletados {len(all_reviews_data)} de {total_reviews} reviews...")
            offset += limit
            if len(all_reviews_data) >= total_reviews:
                break
            time.sleep(0.5)
        except Exception as e:
            st.error(f"Erro ao coletar reviews: {str(e)}")
            break
    if not all_reviews_data:
        return pd.DataFrame()
    return pd.DataFrame(all_reviews_data)

def analisar_dados(df, analisador, batch_size=16):
    """Aplica o modelo de IA e retorna o DataFrame enriquecido."""
    if df.empty or 'comentario' not in df.columns:
        st.error("O DataFrame está vazio ou não contém a coluna 'comentario'.")
        return pd.DataFrame()

    df['sentimento_label'] = 'neutral'
    df['comentario'] = df['comentario'].astype(str).fillna('Este review não possui texto.')
    mask_validos = df['comentario'] != 'Este review não possui texto.'
    comentarios_validos = df.loc[mask_validos, 'comentario'].tolist()

    if not comentarios_validos:
        st.warning("Nenhum comentário válido encontrado para análise.")
        return df

    resultados = []
    progress_bar = st.progress(0, text="Analisando comentários...")
    total_comentarios = len(comentarios_validos)
    total_batches = (total_comentarios + batch_size - 1) // batch_size

    for i in range(0, total_comentarios, batch_size):
        batch = comentarios_validos[i:i+batch_size]
        try:
            batch_results = analisador(batch)
            resultados.extend(batch_results)
            # CORREÇÃO: Garante que o valor do progresso não ultrapasse 1.0
            progress_value = min(i + batch_size, total_comentarios) / total_comentarios
            progress_bar.progress(progress_value, text=f"Analisando lote {i//batch_size + 1}/{total_batches}")
        except Exception as e:
            st.error(f"Erro ao analisar lote: {str(e)}")
            continue
    
    progress_bar.empty()

    def mapear_sentimento(resultado):
        score = int(resultado['label'].split(' ')[0])
        if score <= 2: return 'negative'
        elif score == 3: return 'neutral'
        else: return 'positive'

    sentimentos = [mapear_sentimento(res) for res in resultados]
    df.loc[mask_validos, 'sentimento_label'] = sentimentos

    if 'nota' in df.columns:
        df.loc[df['nota'] >= 4, 'sentimento_label'] = 'positive'
        df.loc[df['nota'] <= 2, 'sentimento_label'] = 'negative'
        df.loc[df['nota'] == 3, 'sentimento_label'] = 'neutral'

    return df

def exibir_dashboard(df):
    """Desenha os componentes visuais do dashboard."""
    st.markdown("---")
    st.subheader("📊 Dashboard de Análise de Sentimentos")

    if df.empty:
        st.warning("Nenhum dado para exibir no dashboard.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total de Itens Analisados", value=len(df))
    
    if 'nota' in df.columns:
        df['nota'] = pd.to_numeric(df['nota'], errors='coerce').fillna(0)
        with col2:
            nota_media = df['nota'].mean()
            st.metric(label="Nota Média (se aplicável)", value=f"{nota_media:.2f} ⭐")
        
        st.markdown("#### Distribuição de Notas (Estrelas)")
        contagem_notas = df['nota'].value_counts().sort_index()
        fig_barras = px.bar(
            contagem_notas, x=contagem_notas.index, y=contagem_notas.values,
            labels={'x': 'Nota (Estrelas)', 'y': 'Quantidade de Avaliações'}, text_auto=True
        )
        fig_barras.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_barras, use_container_width=True)

    st.markdown("#### Análise de Sentimento (Baseada no Texto)")
    contagem_sentimentos = df['sentimento_label'].value_counts()
    fig_pizza = px.pie(
        contagem_sentimentos, values=contagem_sentimentos.values, names=contagem_sentimentos.index,
        title='Distribuição de Sentimentos (Análise de IA)',
        color=contagem_sentimentos.index,
        color_discrete_map={'positive':'#2ca02c', 'negative':'#d62728', 'neutral':'#1f77b4'}
    )
    st.plotly_chart(fig_pizza, use_container_width=True)
    
    st.subheader("💬 Amostra de Comentários por Categoria")
    for sentimento in ['positive', 'negative', 'neutral']:
        df_sentimento = df[df['sentimento_label'] == sentimento]
        if not df_sentimento.empty:
            with st.expander(f"Ver comentários '{sentimento.capitalize()}' ({len(df_sentimento)})"):
                for _, row in df_sentimento.head().iterrows():
                    nota_str = f"({row['nota']} ⭐)" if 'nota' in row and pd.notna(row['nota']) else ""
                    st.write(f"- {row['comentario']} {nota_str}")

# --- Interface Principal da Aplicação ---

st.set_page_config(layout="wide", page_title="Analisador de Sentimentos")
st.title("🤖 Analisador de Sentimentos Multi-fonte")
st.markdown("Analise sentimentos a partir de uma **URL** do Reclame Aqui / Mercado Livre ou de um **arquivo CSV**.")

analisador = carregar_modelo()

if 'resultado_df' not in st.session_state:
    st.session_state.resultado_df = pd.DataFrame()

tab_url, tab_csv = st.tabs(["Analisar por URL", "Analisar por Arquivo CSV"])

with tab_url:
    st.header("Análise a partir de um site")
    with st.form("url_form"):
        url_input = st.text_input("URL do site para análise", placeholder="https://produto.mercadolivre.com.br/MLB-...")
        token_input = st.text_input(
            "Seu Access Token do Mercado Livre (Obrigatório para análise do ML)", 
            type="password", help="Seu token de acesso da aplicação que você criou no Mercado Livre."
        )
        submitted_url = st.form_submit_button("Analisar URL!")

    if submitted_url:
        if url_input:
            with st.spinner("Iniciando análise da URL... Isso pode levar alguns minutos. ☕"):
                df_coletado = pd.DataFrame()
                if "mercadolivre.com.br" in url_input.lower():
                    st.info("Site do Mercado Livre detectado.")
                    if not token_input:
                        st.error("Por favor, insira seu Access Token do Mercado Livre para continuar.")
                    else:
                        id_produto = extrair_id_produto_ml(url_input)
                        if id_produto:
                            df_coletado = coletar_reviews_ml(id_produto, token_input)
                        else:
                            st.error("Não foi possível encontrar um ID de produto (Ex: MLB123456) na URL do Mercado Livre.")
                elif "reclameaqui.com.br" in url_input.lower():
                    st.info("Site do Reclame Aqui detectado.")
                    df_coletado = coletar_reviews_ra(url_input)
                else:
                    st.error("URL não suportada. Insira um link do Reclame Aqui ou Mercado Livre.")

                if not df_coletado.empty:
                    st.info(f"Analisando {len(df_coletado)} reviews coletados...")
                    st.session_state.resultado_df = analisar_dados(df_coletado.copy(), analisador)
                elif url_input:
                    if not ("mercadolivre.com.br" in url_input.lower() and not token_input):
                        st.warning("Coleta finalizada, mas nenhum review foi encontrado.")
                    st.session_state.resultado_df = pd.DataFrame()
        else:
            st.error("Por favor, insira uma URL para análise.")

with tab_csv:
    st.header("Análise a partir de um arquivo")
    st.info("Faça o upload de um arquivo CSV. Ele deve conter uma coluna com os textos a serem analisados.")
    
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    
    if uploaded_file is not None:
        separador = st.selectbox(
            "1. Qual é o separador de colunas do seu arquivo?",
            options=[',', ';', '\t'],
            index=1,
            help="Escolha o caractere que separa as colunas. Ponto e vírgula (;) é comum em arquivos Excel salvos em português."
        )
        
        try:
            df_preview = pd.read_csv(uploaded_file, sep=separador, nrows=5)
            st.write("Pré-visualização dos dados:")
            st.dataframe(df_preview)

            uploaded_file.seek(0)

            colunas_disponiveis = df_preview.columns.tolist()
            coluna_texto = st.selectbox("2. Selecione a coluna que contém os comentários:", colunas_disponiveis)

            if st.button("Analisar Arquivo CSV"):
                with st.spinner("Lendo e analisando o arquivo CSV... ☕"):
                    df_csv = pd.read_csv(uploaded_file, sep=separador)
                    
                    if coluna_texto in df_csv.columns:
                        df_para_analise = df_csv.rename(columns={coluna_texto: 'comentario'})
                        st.session_state.resultado_df = analisar_dados(df_para_analise, analisador)
                    else:
                        st.error(f"A coluna '{coluna_texto}' não foi encontrada no arquivo. Verifique o nome da coluna.")
                        st.session_state.resultado_df = pd.DataFrame()
        except Exception as e:
            st.error(f"Ocorreu um erro ao processar o arquivo CSV: {e}")
            st.info("Dica: Verifique se o separador (vírgula, ponto e vírgula, etc.) está correto e se o arquivo não está vazio ou corrompido.")
            st.session_state.resultado_df = pd.DataFrame()

if st.session_state.resultado_df is not None and not st.session_state.resultado_df.empty:
    exibir_dashboard(st.session_state.resultado_df)