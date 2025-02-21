import os
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import time
import spacy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Baixa as stopwords sem mensagens de log
nltk.download('stopwords', quiet=True)
# Carrega o modelo spaCy para português
nlp = spacy.load("pt_core_news_sm")

# Cria a lista de stopwords combinada (NLTK)
stopwords_pt = set(stopwords.words('portuguese'))

def filtrar_textos(textos):
    """Processa uma lista de textos e retorna uma lista com os textos filtrados."""
    textos_filtrados = []
    with nlp.disable_pipes("ner", "parser"):
        for doc in tqdm(
                nlp.pipe(textos, batch_size=150, n_process=5),
                total=len(textos),
                desc="Filtrando tokens",
                bar_format="{desc}: [Elapsed: {elapsed}] [{n_fmt}/{total_fmt}]"
        ):
            tokens = [token.lemma_.lower() for token in doc
                      if token.pos_ in ["NOUN", "ADJ", "VERB"]
                      and not token.is_stop
                      and len(token.text) > 2]
            textos_filtrados.append(" ".join(tokens))
    return textos_filtrados


def parse_timestamp_list(timestamp_str):
    """Converte uma string com múltiplos timestamps (em ms) para uma lista de objetos datetime."""
    if pd.isna(timestamp_str):
        return []
    try:
        return [pd.to_datetime(int(ts.strip()), unit='ms', errors='coerce')
                for ts in timestamp_str.split(',') if ts.strip()]
    except Exception as e:
        print(f"Erro ao converter timestamp: {e}")
        return []

def parse_history(history_str):
    """Converte uma string de IDs separados por vírgula em uma lista."""
    if pd.isna(history_str):
        return []
    return [item.strip() for item in history_str.split(',') if item.strip()]


def main():
    # ====================================
    # Leitura e processamento dos arquivos de treino
    # ====================================
    TRAIN_DIR = 'treino'
    TRAIN_FILES = [f"treino_parte{i}.csv" for i in range(1, 7)]
    dfs = []

    for file in tqdm(TRAIN_FILES, desc="Arquivos de Treino",
                     bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [Elapsed: {elapsed}] [{n_fmt}/{total_fmt}]"):
        path = os.path.join(TRAIN_DIR, file)
        if os.path.exists(path):
            chunks = []
            for chunk in pd.read_csv(path, chunksize=100000):
                chunk['history_list'] = chunk['history'].apply(parse_history)
                chunk['timestampHistory_list'] = chunk['timestampHistory'].apply(parse_timestamp_list)
                chunk['timestampHistory_new_list'] = chunk['timestampHistory_new'].apply(parse_timestamp_list)
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            dfs.append(df)
        else:
            print(f"Aviso: arquivo não encontrado: {path}")

    if not dfs:
        raise Exception("Nenhum arquivo de treino encontrado!")
    train_df = pd.concat(dfs, ignore_index=True)

    # ====================================
    # Leitura e processamento dos arquivos de itens
    # ====================================
    ITENS_DIR = 'itens'
    ITENS_FILES = [f"itens-parte{i}.csv" for i in range(1, 4)]
    itens_dfs = []
    col_names = ['Page', 'Url', 'Issued', 'Modified', 'Title', 'Body', 'Caption']

    for file in tqdm(ITENS_FILES, desc="Arquivos de Itens",
                     bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [Elapsed: {elapsed}] [{n_fmt}/{total_fmt}]"):
        path = os.path.join(ITENS_DIR, file)
        if os.path.exists(path):
            df = pd.read_csv(path, sep=',', header=None, names=col_names)
            itens_dfs.append(df)
        else:
            print(f"Aviso: arquivo não encontrado: {path}")

    if not itens_dfs:
        raise Exception("Nenhum arquivo de itens encontrado!")
    itens_df = pd.concat(itens_dfs, ignore_index=True)
    print("Leitura dos arquivos de itens concluída.")
    print("Colunas dos itens:", itens_df.columns.tolist())
    print("Algumas linhas dos itens:")
    print(itens_df.head())

    # ====================================
    # Processamento dos dados dos itens
    # ====================================
    print("Processando dados dos itens...")
    date_format = "%Y-%m-%d %H:%M:%S%z"
    itens_df['Issued'] = pd.to_datetime(itens_df['Issued'], format=date_format, errors='coerce')
    itens_df['Modified'] = pd.to_datetime(itens_df['Modified'], format=date_format, errors='coerce')
    reference_date = itens_df['Issued'].max()
    itens_df['age_days'] = (reference_date - itens_df['Issued']).dt.days
    decay = 0.1  # Parâmetro de decaimento para penalizar itens antigos
    itens_df['recency_weight'] = np.exp(-decay * itens_df['age_days'])
    itens_df['text'] = itens_df[['Title', 'Body', 'Caption']].fillna('').agg(' '.join, axis=1)
    print("Aplicando filtragem de tokens nos textos dos itens...")
    itens_df['text_filtrado'] = filtrar_textos(itens_df['text'].tolist())
    print("Filtragem concluída.")

    # ====================================
    # Treinamento do modelo de conteúdo com TF-IDF
    # ====================================
    print("Treinando o vetor TF-IDF. Esse processo pode demorar...")
    portuguese_stop = list(stopwords_pt)
    tfidf = TfidfVectorizer(stop_words=portuguese_stop, token_pattern=r"(?u)\b\w\w+\b", max_features=2000, ngram_range=(1,2))
    start_time = time.time()
    tfidf_matrix = tfidf.fit_transform(itens_df['text_filtrado'])
    tfidf_matrix = tfidf_matrix.multiply(itens_df['recency_weight'].values.reshape(-1, 1))
    tfidf_matrix = tfidf_matrix.tocsr()
    elapsed_time = time.time() - start_time
    print(f"Treinamento do TF-IDF concluído em {elapsed_time:.2f} segundos.")

    article_id_to_idx = {str(row['Page']): idx for idx, row in itens_df.iterrows()}

    # ====================================
    # Salvamento do modelo
    # ====================================
    print("Salvando Modelo...")
    output_path = os.path.join("app", "artifacts", "model.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'train_df': train_df,
            'itens_df': itens_df,
            'tfidf': tfidf,
            'tfidf_matrix': tfidf_matrix,
            'article_id_to_idx': article_id_to_idx,
        }, f)
    print(f"Modelo treinado e salvo em {output_path}!")

    # ====================================
    # Visualizações com matplotlib
    # ====================================
    print("Gerando gráficos com matplotlib...")

    print("Métricas do modelo:")
    print("Número de registros de treino:", len(train_df))
    print("Número de itens:", len(itens_df))
    print("Forma da matriz TF-IDF:", tfidf_matrix.shape)
    total_elements = tfidf_matrix.shape[0] * tfidf_matrix.shape[1]
    sparsity = (tfidf_matrix.nnz / total_elements) * 100
    print(f"Sparsity da matriz TF-IDF: {sparsity:.2f}%")

    # 1. Distribuição da idade dos itens
    plt.figure(figsize=(8, 6))
    plt.hist(itens_df['age_days'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Idade dos itens (dias)")
    plt.ylabel("Frequência")
    plt.title("Distribuição da idade dos itens")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # 2. Distribuição do peso de recência dos itens
    plt.figure(figsize=(8, 6))
    plt.hist(itens_df['recency_weight'].dropna(), bins=30, color='salmon', edgecolor='black')
    plt.xlabel("Peso de recência")
    plt.ylabel("Frequência")
    plt.title("Distribuição do peso de recência dos itens")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # 3. Relação entre idade e peso de recência (scatter plot)
    plt.figure(figsize=(8, 6))
    plt.scatter(itens_df['age_days'], itens_df['recency_weight'], alpha=0.5, color='green')
    plt.xlabel("Tempo em dias dos itens")
    plt.ylabel("Peso de recência")
    plt.title("Decaimento do peso de recência com os dias")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # 4. Distribuição dos valores do TF-IDF
    tfidf_values = tfidf_matrix.data
    plt.figure(figsize=(8, 6))
    plt.hist(tfidf_values, bins=50, color='purple', edgecolor='black')
    plt.xlabel("Valor do TF-IDF")
    plt.ylabel("Frequência")
    plt.title("Distribuição dos valores do TF-IDF")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # 5. Top 20 termos com menor IDF (mais frequentes)
    idf = tfidf.idf_
    vocab = tfidf.get_feature_names_out()
    sorted_indices = np.argsort(idf)
    top_n = 20
    top_words = vocab[sorted_indices][:top_n]
    top_idf = idf[sorted_indices][:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(top_words, top_idf, color='teal')
    plt.xlabel("Valor IDF")
    plt.title("Top 20 termos com menor IDF (mais frequentes)")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    print("Treinamento e visualizações concluídos!")


if __name__ == '__main__':
    main()