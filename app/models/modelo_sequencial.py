import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Configuração do logging
logging.basicConfig(
    filename="treinamento.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def treinar_modelo(df: pd.DataFrame, save_path="modelo_lstm"):
    try:
        logging.info("Iniciando pré-processamento dos dados.")

        # Adicionar EarlyStopping
        early_stopping = EarlyStopping(
            monitor="val_loss",  # Pode ser 'val_accuracy' se preferir
            patience=3,  # Aguardar 3 epochs sem melhoria
            restore_best_weights=True,  # Restaura os melhores pesos do modelo
            verbose=1
        )

        # Garantir que 'history' seja lista
        df['history'] = df['history'].apply(lambda x: x.split(',') if isinstance(x, str) else x)

        # Criar Tokenizer e converter strings em números
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df['history'])
        sequences = tokenizer.texts_to_sequences(df['history'])

        # Definir tamanho fixo para as sequências
        max_length = 10
        sequences_padded = pad_sequences(sequences, maxlen=max_length, padding='post')

        # Criar rótulos fictícios para treinamento (0 ou 1)
        y = np.random.randint(0, 2, len(df))

        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(sequences_padded, y, test_size=0.2, random_state=42)

        logging.info("Dados pré-processados com sucesso.")

        # Criar modelo LSTM
        vocab_size = len(tokenizer.word_index) + 1  # Tamanho do vocabulário
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=2, input_length=max_length),
            LSTM(2, return_sequences=False),
            Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Criar diretório para salvar o modelo
        os.makedirs(save_path, exist_ok=True)
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(save_path, "melhor_modelo.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )

        # Treinar modelo com EarlyStopping
        logging.info("Iniciando treinamento do modelo.")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=15,  # Reduzido para 15 para evitar treinamento excessivo
            batch_size=16,
            callbacks=[model_checkpoint, early_stopping]  # Adicionar EarlyStopping como callback
        )

        # Avaliação do modelo
        loss, accuracy = model.evaluate(X_test, y_test)
        logging.info(f"Acurácia final no conjunto de teste: {accuracy:.4f}")

        # Salvar tokenizer
        with open(os.path.join(save_path, "tokenizer.json"), "w") as f:
            json.dump(tokenizer.word_index, f)

        logging.info(f"Modelo e tokenizer salvos em {save_path}.")

    except Exception as e:
        logging.error(f"Erro durante o treinamento: {str(e)}")

