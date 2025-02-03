import os
import pandas as pd

def carregar_dados_treino(path_treino):
    """
    Carrega todos os arquivos de treino em um único DataFrame.

    Args:
        path_treino (str): Caminho para a pasta contendo os arquivos de treino.

    Returns:
        pd.DataFrame: DataFrame combinado com os dados de treino.
    """
    arquivos_treino = [os.path.join(path_treino, f) for f in os.listdir(path_treino) if f.startswith("treino_parte_")]
    lista_dataframes = []

    for arquivo in arquivos_treino:
        print(f"Carregando {arquivo}...")
        df = pd.read_csv(arquivo)
        lista_dataframes.append(df)

    df_treino = pd.concat(lista_dataframes, ignore_index=True)
    return df_treino

def carregar_dados_itens(path_itens):
    """
    Carrega todos os arquivos de itens em um único DataFrame.

    Args:
        path_itens (str): Caminho para a pasta contendo os arquivos de itens.

    Returns:
        pd.DataFrame: DataFrame combinado com os dados de itens.
    """
    arquivos_itens = [os.path.join(path_itens, f) for f in os.listdir(path_itens) if f.startswith("itens_parte")]
    lista_dataframes = []

    for arquivo in arquivos_itens:
        print(f"Carregando {arquivo}...")
        df = pd.read_csv(arquivo)
        lista_dataframes.append(df)

    df_itens = pd.concat(lista_dataframes, ignore_index=True)
    return df_itens

def preprocessar_dados(df_treino, df_itens):
    """
    Realiza o pré-processamento inicial dos dados de treino e itens.

    Args:
        df_treino (pd.DataFrame): Dados de treino.
        df_itens (pd.DataFrame): Dados de itens.

    Returns:
        pd.DataFrame, pd.DataFrame: DataFrame de treino e DataFrame mesclado.
    """
    # Converter colunas de histórico em listas
    for col in ['history', 'TimestampHistory', 'timeOnPageHistory', 'numberOfClicksHistory',
                'scrollPercentageHistory', 'pageVisitsCountHistory']:
        df_treino[col] = df_treino[col].apply(eval)

    # Expandir o histórico do usuário
    df_treino_explodido = df_treino.explode('history')

    # Mesclar informações das notícias
    df_merged = df_treino_explodido.merge(df_itens, left_on='history', right_on='Page', how='left')

    return df_treino, df_merged

def preprocess_data(history):
    """
    Simula o pré-processamento para recomendações.
    Substitua isso pela lógica real.
    """
    # Por exemplo: recomendar os próximos 5 itens não vistos
    all_items = list(range(1, 101))  # IDs de exemplo
    recommendations = [item for item in all_items if item not in history][:5]
    return recommendations
