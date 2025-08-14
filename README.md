# 🤖 Analisador de Sentimentos Multi-fonte

Este é um aplicativo web construído com **Streamlit** e **Python** que realiza análise de sentimentos em textos coletados de diferentes fontes.  
A análise é feita utilizando um modelo do **Hugging Face Transformers** pré-treinado para análise de sentimentos multilíngue.

## 📊 Modos de Análise
Atualmente, a aplicação suporta a análise de dados de duas maneiras:

- **Via URL**: Coleta e analisa reviews de páginas de produtos do **Mercado Livre** ou páginas de empresas do **Reclame Aqui**.
- **Via Arquivo CSV**: Permite o upload de um arquivo `.csv`, onde o usuário pode selecionar a coluna contendo os textos para análise.

---

## 🚀 Funcionalidades

- **Dashboard Interativo**: Visualização dos dados com gráficos de pizza e barras gerados com **Plotly**.
- **Análise de Múltiplas Fontes**: Suporte para URLs do Mercado Livre, Reclame Aqui e upload de arquivos CSV.
- **Modelo de IA**: Utiliza o modelo `nlptown/bert-base-multilingual-uncased-sentiment` para classificar textos como positivos, negativos ou neutros.
- **Cache Inteligente**: Otimiza o desempenho e o uso de recursos, armazenando em cache o modelo de IA e os dados coletados de URLs.
- **Interface Amigável**: Interface limpa e intuitiva, organizada em abas para cada tipo de análise.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Streamlit**: Para a criação da interface web.
- **Pandas**: Para manipulação e estruturação dos dados.
- **Plotly Express**: Para a criação dos gráficos interativos.
- **Hugging Face Transformers**: Para carregar e utilizar o modelo de análise de sentimentos.
- **PyTorch** ou **TensorFlow**: Como backend para o modelo da Hugging Face.
- **Beautiful Soup 4** e **Requests**: Para web scraping da página do Reclame Aqui.

---

## 📋 Pré-requisitos

Antes de começar, certifique-se de que você tem o **Python 3.8 ou superior** instalado em sua máquina.

---

## ⚙️ Configuração e Instalação

### 1. Clone o Repositório
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Crie e Ative um Ambiente Virtual (venv)
É uma boa prática usar um ambiente virtual para isolar as dependências do projeto.

No Windows:

```bash
python -m venv venv

.\venv\Scripts\activate
```

No macOS e Linux:
```bash
python3 -m venv venv

source venv/bin/activate
```
Após a ativação, você verá (venv) no início do seu terminal.

### 3. Crie o arquivo requirements.txt
Este arquivo lista todas as bibliotecas Python que o projeto precisa.
Crie um arquivo chamado requirements.txt na raiz do seu projeto com o seguinte conteúdo:

```shell
streamlit

pandas

plotly-express

requests

beautifulsoup4

transformers

torch

\# ou tensorflow, dependendo do backend que preferir
```

### 4. Instale as Dependências
Com o ambiente virtual ativado, instale todas as bibliotecas listadas no `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 🔑 Configuração Adicional (Mercado Livre)

Para analisar produtos do Mercado Livre, você precisará de um Access Token.

1. Crie uma aplicação no site de desenvolvedores do Mercado Livre.

2. Após criar a aplicação, você terá acesso ao seu Access Token.

3. Este token deverá ser inserido no campo correspondente na interface do aplicativo ao analisar uma URL do Mercado Livre.

### ▶️ Como Executar a Aplicação

Com o ambiente virtual ativado e as dependências instaladas, execute o seguinte comando no seu terminal:
```bash
streamlit run seu\_arquivo\_principal.py
```
Substitua `seu\_arquivo\_principal.py` pelo nome do seu script Python (ex: `app.py`).

A aplicação será aberta automaticamente no seu navegador padrão.

### 📖 Como Usar

Análise por URL

1. Selecione a aba "Analisar por URL".

2. Cole a URL de uma página de empresa do Reclame Aqui ou de um produto do Mercado Livre.

3. Se for uma URL do Mercado Livre, preencha o campo do Access Token.

4. Clique em "Analisar URL!" e aguarde o resultado.

### Análise por Arquivo CSV

1. Selecione a aba "Analisar por Arquivo CSV".

2. Clique em "Escolha um arquivo CSV" para fazer o upload do seu arquivo.

3. Selecione o separador de colunas correto (, ou ;). Uma pré-visualização será exibida.

4. Selecione a coluna que contém os textos a serem analisados.

5. Clique em "Analisar Arquivo CSV" e aguarde o dashboard ser gerado.


### 🌐 Acesse o Aplicativo Online

O **Analisador de Sentimentos Multi-fonte** já está disponível para uso direto no seu navegador, sem precisar instalar nada localmente.

---

## 🚀 Link de Acesso

🔗 **[Clique aqui para abrir o aplicativo](https://customer-sentiment-analyzer-lucasmartinssw.streamlit.app/)**

---
