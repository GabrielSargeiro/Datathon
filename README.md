## Deploy

Veja [DEPLOY.md](DEPLOY.md) para instruções de build, push e deploy na AWS.

## Estrutura do projeto

    Datathon/
    ├── app/
    │   ├── main.py               # Entrada da API FastAPI
    │   ├── routers/
    │   │   └── recommendation.py  # Endpoints da API
    │   ├── models/
    │   │   └── data_models.py     # Schemas de dados (Pydantic)
    │   ├── services/
    │   │   ├── model.py           # Carregamento e inferência do model.pkl
    │   │   └── preprocessing.py   # Funções de pré-processamento de dados (A MUDAR)
    │   └── utils/
    │       └── helpers.py         # Funções auxiliares
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── itens/                    # Dados dos itens
    │   ├── itens-parte1.csv
    │   ├── itens-parte2.csv
    │   └── itens-parte3.csv
    ├── treino/                   # Dados de treinamento
    │   ├── treino_parte1.csv
    │   ├── treino_parte2.csv
    │   ├── treino_parte3.csv
    │   ├── treino_parte4.csv
    │   ├── treino_parte5.csv
    │   └── treino_parte6.csv
    ├── .env
    ├── train_model.py            # Script para treinar e gerar o model.pkl (TEMPORARIO)
    ├── model.pkl                 # Modelo gerado pelo script de treinamento
    ├── build_and_push.sh
    ├── DEPLOY.md
    ├── README.md
    └── requirements.txt

