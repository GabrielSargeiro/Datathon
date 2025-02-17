import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# --- Leitura dos arquivos de treino ---
TRAIN_DIR = '../treino'
TRAIN_FILES = [f"treino_parte{i}.csv" for i in range(1, 7)]
dfs = []
for file in TRAIN_FILES:
    path = os.path.join(TRAIN_DIR, file)
    if os.path.exists(path):
        print(f"Lendo {path}...")
        dfs.append(pd.read_csv(path))
    else:
        print(f"Aviso: arquivo não encontrado: {path}")

if not dfs:
    raise Exception("Nenhum arquivo de treino encontrado!")
train_df = pd.concat(dfs, ignore_index=True)

# Converter a coluna 'history' em lista (IDs separados por vírgula)
def parse_history(history_str):
    if pd.isna(history_str):
        return []
    return [item.strip() for item in history_str.split(',') if item.strip()]

train_df['history_list'] = train_df['history'].apply(parse_history)

# --- Leitura dos arquivos de itens ---
ITENS_DIR = '../itens'
ITENS_FILES = [f"itens-parte{i}.csv" for i in range(1, 4)]
itens_dfs = []

# Define os nomes das colunas esperadas para os itens
col_names = ['Page', 'Url', 'Issued', 'Modified', 'Title', 'Body', 'Caption']

for file in ITENS_FILES:
    path = os.path.join(ITENS_DIR, file)
    if os.path.exists(path):
        print(f"Lendo {path}...")
        # Força a leitura com vírgula como separador, sem cabeçalho e atribuindo os nomes definidos
        df = pd.read_csv(path, sep=',', header=None, names=col_names)
        itens_dfs.append(df)
    else:
        print(f"Aviso: arquivo não encontrado: {path}")

if not itens_dfs:
    raise Exception("Nenhum arquivo de itens encontrado!")
itens_df = pd.concat(itens_dfs, ignore_index=True)

# Exibe as colunas e algumas linhas para confirmar a leitura correta dos dados
print("Colunas dos itens:", itens_df.columns.tolist())
print("Algumas linhas dos itens:")
print(itens_df.head())
print("Carregando...")

# Cria uma coluna 'text' que concatena Title, Body e Caption (ajustando para valores nulos)
itens_df['text'] = itens_df[['Title', 'Body', 'Caption']].fillna('').agg(' '.join, axis=1)

# --- Construção do modelo de conteúdo ---
# Obtém as stop words para o português usando NLTK
portuguese_stop = stopwords.words('portuguese')

# Cria o vetor TF-IDF usando as stop words em português e define o número máximo de features
tfidf = TfidfVectorizer(stop_words=portuguese_stop, max_features=5000)
tfidf_matrix = tfidf.fit_transform(itens_df['text'])

# Cria um mapeamento do id do artigo (campo "Page") para o índice na matriz TF-IDF
article_id_to_idx = {str(row['Page']): idx for idx, row in itens_df.iterrows()}

# Salva os objetos do modelo em um arquivo pickle
model_data = {
    'train_df': train_df,
    'itens_df': itens_df,
    'tfidf': tfidf,
    'tfidf_matrix': tfidf_matrix,
    'article_id_to_idx': article_id_to_idx,
}

with open('../model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Modelo treinado e salvo em model.pkl!")
