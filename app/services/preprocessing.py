import pandas as pd
import os


caminho = os.getcwd()

caminho = os.path.dirname(os.path.dirname(caminho))

def consolidar_treinos() -> None:

    if not os.path.isfile(caminho + r'/treino/consolidado.csv'):
        arquivos = [arquivo for arquivo in os.listdir(caminho + r"/treino/") if arquivo.endswith('.csv')]
        df_final = []
        for arquivo in arquivos:
            df = pd.read_csv(caminho + fr"/treino/{arquivo}", sep=',')
            df_final.append(df)

        dataframe = pd.concat(df_final, ignore_index=True)
        dataframe.to_csv(caminho + r'/treino/consolidado.csv', index=False, sep=';')


def consolidar_itens() -> None:
    if not os.path.isfile(caminho + r'/treino/consolidado_itens.csv'):
        arquivos = [arquivo for arquivo in os.listdir(caminho + r"/itens/") if arquivo.endswith('.csv')]
        df_final = []
        for arquivo in arquivos:
            df = pd.read_csv(caminho + fr"/itens/{arquivo}", sep=',')
            df_final.append(df)

        dataframe = pd.concat(df_final, ignore_index=True)
        dataframe.to_csv(caminho + r'/treino/consolidado_itens.csv', index=False, sep=';')


def cruzar_history_materia():
    dataframe = pd.read_csv(caminho + r'/treino/consolidado_itens.csv', sep=';')
    df = pd.read_csv(caminho + r'/treino/consolidado.csv', sep=';')
    result = pd.merge(df, dataframe[['page', 'url']], left_on='history', right_on='page', how='left')
    result.dropna(axis=0, inplace=True)
    result.to_csv(caminho + r'/treino/consolidado_url.csv', sep=';', index=False)

