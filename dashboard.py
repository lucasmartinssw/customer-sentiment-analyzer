import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title('📊 Painel de Análise de Sentimento de Clientes')

# Carregar os dados analisados
try:
    df = pd.read_csv('reviews_analisados.csv')
except FileNotFoundError:
    st.error("Arquivo 'reviews_analisados.csv' não encontrado. Execute a análise primeiro.")
    st.stop()

# --- Layout do Dashboard ---
# 1. Resumo Geral
st.subheader("Resumo Geral dos Sentimentos")
contagem_sentimentos = df['sentimento_label'].value_counts()
st.dataframe(contagem_sentimentos)

# 2. Gráfico de Pizza
fig_pizza = px.pie(df, names='sentimento_label', title='Distribuição de Sentimentos', 
                     color='sentimento_label',
                     color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'blue'})

st.plotly_chart(fig_pizza, use_container_width=True)

# 3. Mostrar os comentários
st.subheader("Amostra dos Comentários")

ver_positivos = st.checkbox('Ver comentários positivos')
if ver_positivos:
    st.write(df[df['sentimento_label'] == 'positive']['comentario'].head())

ver_negativos = st.checkbox('Ver comentários negativos')
if ver_negativos:
    st.write(df[df['sentimento_label'] == 'negative']['comentario'].head())

# NOTA: A exibição dos tópicos pode ser adicionada aqui, lendo os resultados da Fase 2.