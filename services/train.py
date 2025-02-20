import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)

from model import treinar_modelo_feature
from preprocessing import consolidar_treinos, consolidar_itens


def main():
    print("Consolidando bases de treino...")
    consolidar_treinos()
    consolidar_itens()

    print("Iniciando treinamento do modelo LightFM...")
    treinar_modelo_feature()

if __name__ == '__main__':
    main()
