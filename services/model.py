import os
import pandas as pd
import joblib
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from sklearn.model_selection import KFold
from scipy.sparse import coo_matrix
import logging
import numpy as np
from preprocessing import consolidar_treinos, consolidar_itens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("treinamento_modelo.log", mode="w")
    ]
)


def filtrar_dados(dataframe):
    # Converte colunas para categoria e formata datas e números
    dataframe['url'] = dataframe['url'].astype("category")
    dataframe['page'] = dataframe['page'].astype("category")
    dataframe['history'] = dataframe['history'].astype("category")
    dataframe['userId'] = dataframe['userId'].astype("category")
    dataframe['timestampHistory'] = pd.to_datetime(dataframe['timestampHistory'], errors='coerce')
    dataframe['pageVisitsCountHistory'] = dataframe['pageVisitsCountHistory'].astype("int64")
    # Normaliza o timestamp para [0,1]
    dataframe["timestampHistory"] = (dataframe["timestampHistory"] - dataframe["timestampHistory"].min()) / (
             dataframe["timestampHistory"].max() - dataframe["timestampHistory"].min())

    # Filtra os registros com pageVisitsCountHistory >= 5
    dataframe = dataframe[dataframe['pageVisitsCountHistory'] >= 5]


    dataframe = dataframe[dataframe['userType'] == 'Logged']

    dataframe.dropna(axis=0, inplace=True)

    return dataframe

caminho = os.getcwd()
def tratamento_dataframe_consolidado():
    caminho_dados_tratados = os.path.join(caminho, "treino", "dados_tratados.csv")

    if not os.path.isfile(caminho_dados_tratados):
        # Lê o arquivo consolidado
        df = pd.read_csv(os.path.join(caminho, "treino", "consolidado.csv"), sep=';')

        colunas = ['history', 'timestampHistory', 'numberOfClicksHistory',
                   'timeOnPageHistory', 'timestampHistory_new',
                   'pageVisitsCountHistory', 'scrollPercentageHistory']
        # Divide as colunas que são strings contendo listas
        for coluna in colunas:
            df[coluna] = df[coluna].apply(lambda x: x.split(',') if isinstance(x, str) else x)

        logging.info("Explodindo a coluna 'history' para múltiplas linhas.")
        dataframe = df.explode('history').reset_index(drop=True)

        # Expande as outras colunas para manter a correspondência
        dataframe = dataframe.assign(
            numberOfClicksHistory=df["numberOfClicksHistory"].explode().values,
            timeOnPageHistory=df["timeOnPageHistory"].explode().values,
            timestampHistory=df["timestampHistory"].explode().values,
            timestampHistory_new=df["timestampHistory_new"].explode().values,
            pageVisitsCountHistory=df["pageVisitsCountHistory"].explode().values,
            scrollPercentageHistory=df['scrollPercentageHistory'].explode().values
        )

        # Converte timestamp para datetime
        dataframe['timestampHistory'] = pd.to_datetime(dataframe['timestampHistory'].astype('int64'), unit='ms')

        # Lê o arquivo de itens e faz merge
        df_itens = pd.read_csv(os.path.join(caminho, "treino", "consolidado_itens.csv"),
                               sep=';')
        dataframe = pd.merge(dataframe, df_itens[['page', 'url']], left_on='history', right_on='page', how='left')

        # Salva os dados processados (consolidados)
        dataframe.to_csv(caminho_dados_tratados, sep=';', index=False)
        dataframe['categoria'] = dataframe['url'].str.extract(r'http://g1.globo.com/([a-zA-Z]+)/')
        return dataframe
    else:
        dataframe = pd.read_csv(caminho_dados_tratados, sep=';')

        dataframe['categoria'] = dataframe['url'].str.extract(r'http://g1.globo.com/([a-zA-Z\-]+)/')

        dataframe = dataframe.fillna('')
        return dataframe


def tratamento_dataframe_filtrado():
    dataframe = tratamento_dataframe_consolidado()
    dataframe_filtrado = filtrar_dados(dataframe)
    return dataframe_filtrado



def treinar_modelo_feature(num_epochs: int = 10, k: int = 5):
    logging.info("Iniciando o treinamento do modelo.")


    dataframe_treino = tratamento_dataframe_filtrado()

    dataframe_teste = tratamento_dataframe_consolidado()


    limite_interacoes = 3
    dataframe_treino['timestampHistory'] = pd.to_datetime(dataframe_treino['timestampHistory'].astype('int64'),
                                                          unit='ms')
    dataframe_treino = dataframe_treino[dataframe_treino['pageVisitsCountHistory'] >= limite_interacoes]
    dataframe_treino = dataframe_treino[['userId', 'url', 'history', 'timestampHistory',
                                        'numberOfClicksHistory', 'historySize', 'categoria']]

    dataframe_teste['timestampHistory'] = pd.to_datetime(dataframe_teste['timestampHistory'], errors='coerce')
    dataframe_teste = dataframe_teste[dataframe_teste['pageVisitsCountHistory'] >= limite_interacoes]
    dataframe_teste = dataframe_teste[['userId', 'url', 'history', 'timestampHistory', 'userType',
                                        'numberOfClicksHistory', 'historySize', 'categoria']]

    # Criar o dataset LightFM utilizando a união dos IDs de treino e teste
    logging.info("Criando o dataset para treinamento e avaliação.")
    dataset = Dataset()
    dataset.fit(
        np.concatenate([dataframe_treino["userId"].unique(), dataframe_teste["userId"].unique()]),
        np.concatenate([dataframe_treino['url'].unique(), dataframe_teste['url'].unique()])
    )

    logging.info("Construindo as interações (dados filtrados para treino).")
    interactions_train, _ = dataset.build_interactions(dataframe_treino[["userId", "url"]].values)
    interactions_train = interactions_train.tocoo()


    logging.info("Construindo as interações (dados consolidados para teste).")
    interactions_test, _ = dataset.build_interactions(dataframe_teste[["userId", "url"]].values)
    interactions_test = interactions_test.tocoo()

    loss = 'warp'
    logging.info(f"Definindo o modelo LightFM com loss {loss}.")
    model = LightFM(loss=loss, learning_rate=0.05, no_components=10, item_alpha=1e-4, user_alpha=1e-4)


    kf = KFold(n_splits=5)
    melhor_precisao = 0
    melhor_modelo = None

    n_nonzero = interactions_train.data.shape[0]
    for train_index, test_index in kf.split(range(n_nonzero)):
        train_interactions = coo_matrix(
            (interactions_train.data[train_index],
             (interactions_train.row[train_index], interactions_train.col[train_index])),
            shape=interactions_train.shape
        )
        val_interactions = coo_matrix(
            (interactions_train.data[test_index],
             (interactions_train.row[test_index], interactions_train.col[test_index])),
            shape=interactions_train.shape
        )

        logging.info(f"Iniciando o treinamento com {num_epochs} épocas.")
        model.fit(train_interactions, epochs=num_epochs, num_threads=os.cpu_count())

        # Avaliação usando os dados consolidados (teste)
        precisao = precision_at_k(model, interactions_test, k=k).mean()
        recall = recall_at_k(model, interactions_test, k=k).mean()
        auc = auc_score(model, interactions_test).mean()

        logging.info(f'Precisão no top-{k}: {precisao:.4f}')
        logging.info(f'Recall no top-{k}: {recall:.4f}')
        logging.info(f'AUC Score: {auc:.4f}')

        if precisao > melhor_precisao:
            melhor_precisao = precisao
            melhor_modelo = model

    if melhor_modelo:
        logging.info("Salvando o modelo treinado com melhor precisão.")
        joblib.dump(melhor_modelo, os.path.join("app", "artifacts", "melhor_modelo_lightfm.pkl"))

    logging.info("Treinamento concluído.")
    return melhor_modelo


if __name__ == "__main__":
    logging.info('Consolidando bases de treino')
    consolidar_treinos()
    consolidar_itens()
    logging.info("Iniciando treino do modelo")
    treinar_modelo_feature()

