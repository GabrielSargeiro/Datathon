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
import string
from concurrent.futures import ProcessPoolExecutor, as_completed

nltk.download('stopwords', quiet=True)
nlp = spacy.load("pt_core_news_sm")

# Cria a lista de stopwords combinada (NLTK)
stopwords_pt = set(stopwords.words('portuguese'))


def filtrar_textos(textos):
    """Processa uma lista de textos e retorna uma lista de strings filtradas."""
    textos_filtrados = []
    with nlp.disable_pipes("ner", "parser"):
        for doc in tqdm(nlp.pipe(textos, batch_size=150, n_process=4),
                        total=len(textos),
                        desc="Filtrando tokens",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}]"):
            tokens = []
            for token in doc:
                if (token.pos_ in ["NOUN", "PROPN"]) or (token.ent_type_ in ["LOC", "ORG", "PER"]):
                    lemma = token.lemma_.lower().strip()
                    if lemma not in stopwords_pt and lemma not in string.punctuation:
                        if len(lemma) > 2:
                            tokens.append(lemma)
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


def read_train_csv_file(path):
    df = pd.read_csv(path, engine="pyarrow")
    df['history_list'] = df['history'].apply(parse_history)
    df['timestampHistory_list'] = df['timestampHistory'].apply(parse_timestamp_list)
    df['timestampHistory_new_list'] = df['timestampHistory_new'].apply(parse_timestamp_list)
    return df


def process_train_files():
    TRAIN_DIR = '../treino'
    TRAIN_FILES = [f"treino_parte{i}.csv" for i in range(1, 7)]
    dfs = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        future_to_file = {
            executor.submit(read_train_csv_file, os.path.join(TRAIN_DIR, file)): file
            for file in TRAIN_FILES if os.path.exists(os.path.join(TRAIN_DIR, file))
        }
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Arquivos de Treino",
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}]"):
            try:
                df = future.result()
                dfs.append(df)
            except Exception as exc:
                file = future_to_file[future]
                print(f"Erro no arquivo {file}: {exc}")
    if not dfs:
        raise Exception("Nenhum arquivo de treino encontrado!")
    return pd.concat(dfs, ignore_index=True)


def main():
    # ==============================================
    # Leitura e processamento dos arquivos de treino (paralelo)
    # ==============================================

    print("Iniciando a leitura dos arquivos de treino...")
    train_df = process_train_files()

    # ==============================================
    # Leitura e processamento dos arquivos de itens
    # ==============================================

    ITENS_DIR = '../itens'
    ITENS_FILES = [f"itens-parte{i}.csv" for i in range(1, 4)]
    itens_dfs = []
    col_names = ['Page', 'Url', 'Issued', 'Modified', 'Title', 'Body', 'Caption']
    for file in tqdm(ITENS_FILES, desc="Arquivos de Itens",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}]"):
        path = os.path.join(ITENS_DIR, file)
        if os.path.exists(path):
            df = pd.read_csv(path, sep=',', header=None, names=col_names)
            itens_dfs.append(df)
        else:
            print(f"Aviso: arquivo não encontrado: {path}")
    if not itens_dfs:
        raise Exception("Nenhum arquivo de itens encontrado!")
    itens_df = pd.concat(itens_dfs, ignore_index=True)

    # ==============================================
    # Processamento dos dados dos itens
    # ==============================================

    date_format = "%Y-%m-%d %H:%M:%S%z"
    itens_df['Issued'] = pd.to_datetime(itens_df['Issued'], format=date_format, errors='coerce')
    itens_df['Modified'] = pd.to_datetime(itens_df['Modified'], format=date_format, errors='coerce')
    reference_date = itens_df['Issued'].max()
    itens_df['age_days'] = (reference_date - itens_df['Issued']).dt.days

    limite_dias = 60
    itens_df = itens_df[itens_df['age_days'] <= limite_dias].copy()
    decay = 0.08
    # Sem categorização pois via palavras-chave seria subjetiva; o ideal seria um campo de categoria.

    itens_df['recency_weight'] = np.exp(-decay * itens_df['age_days'])
    itens_df['text'] = itens_df[['Title', 'Body', 'Caption']].fillna('').agg(' '.join, axis=1)
    itens_df['text_filtrado'] = filtrar_textos(itens_df['text'].tolist())

    # ==============================================
    # Treinamento do modelo de conteúdo com TF-IDF
    # ==============================================

    print("Treinando o vetor TF-IDF...")
    portuguese_stop = list(stopwords_pt)
    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=(1, 2),
        max_features=3000,
        min_df=3,
        max_df=0.75,
        stop_words=portuguese_stop
    )
    start_time = time.time()
    tfidf_matrix = tfidf.fit_transform(itens_df['text_filtrado'])
    tfidf_matrix = tfidf_matrix.multiply(itens_df['recency_weight'].values.reshape(-1, 1))
    tfidf_matrix = tfidf_matrix.tocsr()
    elapsed_time = time.time() - start_time
    print(f"Treinamento do TF-IDF concluído em {elapsed_time:.2f} segundos.")

    article_id_to_idx = {str(row['Page']): idx for idx, row in itens_df.iterrows()}

    # ==============================================
    # Salvamento do modelo TF-IDF
    # ==============================================

    print("Salvando Modelo TF-IDF...")
    output_path = os.path.join("../app", "artifacts", "model.pkl")
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

    # ==============================================
    # Visualizações com matplotlib
    # ==============================================

    print("==============================================")
    print("Gerando gráficos com matplotlib...")
    print("Número de registros de treino:", len(train_df))
    print("Número de itens:", len(itens_df))
    print("Forma da matriz TF-IDF:", tfidf_matrix.shape)
    total_elements = tfidf_matrix.shape[0] * tfidf_matrix.shape[1]
    sparsity = (tfidf_matrix.nnz / total_elements) * 100
    print(f"Sparsity da matriz TF-IDF: {sparsity:.2f}%")

    # Gráfico 1: Distribuição da idade dos itens
    plt.figure(figsize=(8, 6))
    plt.hist(itens_df['age_days'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Idade dos itens (dias)")
    plt.ylabel("Frequência")
    plt.title("Distribuição da idade dos itens")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Gráfico 2: Relação entre idade e peso de recência (scatter plot)
    plt.figure(figsize=(8, 6))
    plt.scatter(itens_df['age_days'], itens_df['recency_weight'], alpha=0.5, color='green')
    plt.xlabel("Tempo em dias dos itens")
    plt.ylabel("Peso de recência")
    plt.title("Decaimento do peso de recência com os dias")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # Gráfico 3: Top 20 termos com menor IDF (mais frequentes)
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

    # ==============================================
    # Exibe o vocabulário completo no console
    # ==============================================

    print("==============================================")
    print("Vocabulário ordenado por peso (IDF) abaixo")
    print("==============================================")
    vocabulario = tfidf.get_feature_names_out()
    idf = tfidf.idf_
    vocab_idf = list(zip(vocabulario, idf))
    vocab_idf_sorted = sorted(vocab_idf, key=lambda x: x[1])
    for token, weight in vocab_idf_sorted:
        print(f"{token}: {weight}")
    print("==============================================")
    print("Vocabulário ordenado por peso (IDF) acima")

    print("==============================================")
    print("Treinamento e visualizações concluídos!")
    print("==============================================")


if __name__ == '__main__':
    main()
