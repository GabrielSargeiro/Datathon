import pandas as pd
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, MultiHeadAttention, Add, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

# tf.config.optimizer.set_jit(False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("treinamento_modelo_sasrec.log", mode="w")
    ]
)




def carregar_dados(max_seq_len=10):
    caminho = os.path.dirname(os.path.dirname(os.getcwd()))
    caminho_dados = os.path.join(caminho, "treino", "dados_tratados.csv")

    if not os.path.isfile(caminho_dados):
        logging.error("Arquivo de dados não encontrado.")
        return None

    try:
        df = pd.read_csv(caminho_dados, sep=';')
    except Exception as e:
        logging.error(f"Erro ao ler o CSV: {e}")
        return None

    logging.info(f"Total de registros carregados: {len(df)}")
    if df.empty:
        logging.error("Dataset vazio após carregar o CSV.")
        return None

    df['categoria'] = df['url'].str.extract(r'http://g1.globo.com/([a-zA-Z]+)/')
    df = df[df['userType'] == 'Logged']
    logging.info(f"Registros após filtrar 'Logged': {len(df)}")
    if df.empty:
        logging.error("Nenhum usuário 'Logged' encontrado.")
        return None

    df = df.dropna(subset=['url'])
    logging.info(f"Registros após remover URLs nulas: {len(df)}")
    if df.empty:
        logging.error("Todos os URLs são nulos após a filtragem.")
        return None

    user_mapping = {user: idx for idx, user in enumerate(df['userId'].unique(), 1)}
    item_mapping = {item: idx for idx, item in enumerate(df['url'].unique(), 1)}

    df['userId'] = df['userId'].map(user_mapping).fillna(0).astype(int)
    df['url'] = df['url'].map(item_mapping).fillna(0).astype(int)

    user_item_sequences = df.groupby('userId')['url'].apply(list).reset_index()

    X_users = []
    X_seqs = []
    y_labels = []

    for _, row in user_item_sequences.iterrows():
        user_id = row['userId']
        seq = row['url']

        if len(seq) < 1:
            continue

        input_seq = seq[:-1] if len(seq) > 1 else seq
        label = seq[-1] if len(seq) > 1 else seq[0]

        padded_seq = pad_sequences(
            [input_seq],
            maxlen=max_seq_len,
            padding='post',
            truncating='pre',
            value=0
        )[0]

        X_users.append(user_id)
        X_seqs.append(padded_seq)
        y_labels.append(label)

    if len(X_users) == 0:
        logging.error("Nenhum dado válido após o pré-processamento.")
        return None

    return (
        np.array(X_users, dtype=np.int32),
        np.array(X_seqs, dtype=np.int32),
        np.array(y_labels, dtype=np.int32),
        len(user_mapping) + 1,
        len(item_mapping) + 1
    )


def criar_sasrec(vocab_size_users, vocab_size_items, max_seq_len=10, embed_dim=64, num_heads=2, ff_dim=128):
    user_input = Input(shape=(1,), dtype=tf.int32, name='user_input')
    seq_input = Input(shape=(max_seq_len,), dtype=tf.int32, name='seq_input')

    # Embeddings
    user_embed = Embedding(vocab_size_users, embed_dim, name='user_embedding')(user_input)
    item_embed = Embedding(vocab_size_items, embed_dim, name='item_embedding')(seq_input)

    # Usar Lambda layer para aplicar tf.tile
    user_embed = Lambda(lambda x: tf.tile(x, [1, max_seq_len, 1]))(user_embed)  # Corrigido!

    # Combinar embeddings
    x = Add()([item_embed, user_embed])
    x = LayerNormalization()(x)

    # Atenção Multi-cabeça
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    # Feed-forward
    ff_output = Dense(ff_dim, activation='relu', kernel_regularizer=l2(0.05))(x)
    ff_output = Dense(embed_dim)(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization()(x)

    # Dropout
    x = Dropout(0.75)(x)

    # Flatten
    x = Flatten()(x)
    output = Dense(vocab_size_items, activation='softmax')(x)  # Softmax para multiclasse

    model = Model(inputs=[user_input, seq_input], outputs=output)
    model.compile(
        optimizer=Adam(0.005),
        loss='sparse_categorical_crossentropy',  # Perda para labels inteiras
        metrics=['sparse_categorical_accuracy']  # Métrica correta
    )
    return model

def treinar_modelo_sasrec(num_epochs=10, batch_size=32, max_seq_len=10):
    logging.info("Iniciando o treinamento do modelo SASRec.")

    data = carregar_dados(max_seq_len)
    if data is None:
        logging.error("Não foi possível carregar os dados. Verifique os logs.")
        return None

    X_users, X_seqs, y_labels, vocab_users, vocab_items = data

    try:
        indices = np.arange(len(X_users))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    except ValueError as e:
        logging.error(f"Erro ao dividir os dados: {e}")
        return None

    model = criar_sasrec(vocab_users, vocab_items, max_seq_len)

    checkpoint = ModelCheckpoint(
        "modelos/melhor_modelo_sasrec.h5",
        monitor='val_sparse_categorical_accuracy',
        save_best_only=True,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    logging.info("Treinando modelo SASRec.")
    model.fit(
        [X_users[train_idx], X_seqs[train_idx]],
        y_labels[train_idx],
        validation_data=([X_users[test_idx], X_seqs[test_idx]], y_labels[test_idx]),
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping]
    )

    logging.info("Treinamento concluído.")
    return model

if __name__ == '__main__':
    treinar_modelo_sasrec()