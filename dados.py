import pandas as pd
import os


caminho = os.getcwd()

def consolidar_treinos() -> None:


    arquivos = [arquivo for arquivo in os.listdir(caminho + fr"/bases/files/treino/") if arquivo.endswith('.csv')]
    df_final = []
    for arquivo in arquivos:
        df = pd.read_csv(caminho + fr"/bases/files/treino/{arquivo}", sep=',')
        df_final.append(df)

    dataframe = pd.concat(df_final, ignore_index=True)

    dataframe.to_csv(caminho + r'/bases/files/resultados/consolidado.csv', index=False, sep=';')


def tratar_dados() -> pd.DataFrame:

    df = pd.read_csv(caminho + r'/bases/files/resultados/consolidado.csv', sep=';')

    # Removendo valores nulos e duplicatas
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df