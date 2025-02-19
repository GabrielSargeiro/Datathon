from modelo_lightfm import treinar_modelo_feature
from modelo_sequencial import treinar_modelo
from dados import tratar_dados, consolidar_treinos


def main():
    consolidar_treinos()
    df = tratar_dados()
    treinar_modelo_feature(df)
    treinar_modelo(df)

if __name__ == "__main__":
    main()
