from lightfm.data import Dataset
import pandas as pd
import logging
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm import cross_validation
import joblib


# Configuração do logger
logging.basicConfig(
    level=logging.INFO,  # Nível de log
    format="%(asctime)s - %(levelname)s - %(message)s",  # Formato de log
    handlers=[
        logging.StreamHandler(),  # Exibe os logs no console
        logging.FileHandler("treinamento_modelo.log", mode="w")  # Salva os logs em um arquivo
    ]
)


def treinar_modelo_feature(df: pd.DataFrame, num_epochs: int = 5, k: int = 5):
    logging.info("Iniciando o treinamento do modelo.")

    # Transformar a coluna 'history' de string para lista
    df['history'] = df['history'].apply(lambda x: x.split(','))

    logging.info("Explodindo a coluna 'history' para múltiplas linhas.")

    # Explodir a coluna 'history' para múltiplas linhas por 'userId'
    dataframe = df.explode('history').reset_index(drop=True)

    # Criar dataset
    logging.info("Criando o dataset para treinamento.")
    dataset = Dataset()
    dataset.fit(dataframe["userId"].unique(), dataframe["history"].unique())

    # Construir interações
    logging.info("Construindo as interações.")
    interactions, _ = dataset.build_interactions(dataframe[["userId", "history"]].values)

    # Definir o modelo
    logging.info("Definindo o modelo LightFM com 'warp' como função de perda.")
    model = LightFM(loss="warp")

    # Realizar Cross-Validation (dividir em 20% para teste)
    cv_results = cross_validation.random_train_test_split(interactions, test_percentage=0.2)
    train_interactions = cv_results[0]
    test_interactions = cv_results[1]

    logging.info(f"Iniciando o treinamento com {num_epochs} épocas.")

    # Treinar o modelo
    model.fit(train_interactions, epochs=num_epochs, num_threads=2)

    # Avaliar a precisão no conjunto de validação
    precision = precision_at_k(model, test_interactions, k=k).mean()
    logging.info(f'Precisão no top-{k}: {precision:.4f}')

    # Salvar o melhor modelo
    logging.info("Salvando o modelo treinado.")
    joblib.dump(model, 'melhor_modelo_lightfm.pkl')

    logging.info("Treinamento concluído e modelo salvo.")

    return model


