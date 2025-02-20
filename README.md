## Deploy

Veja [DEPLOY.md](DEPLOY.md) para instruções de build, push e deploy na AWS.

## Estrutura do projeto

    Datathon/
    ├── app/
    │   ├── config.py                      # Configurações da API
    │   ├── main.py                        # Ponto de entrada da API FastAPI
    │   ├── artifacts/
    │   │   └── melhor_modelo_lightfm.pkl  # Modelo treinado para inferência
    │   ├── models/
    │   │   └── data_models.py             # Schemas de dados (Pydantic)
    │   └── routers/
    │       └── recommendation.py          # Endpoints da API
    ├── services/                          # Scripts de treinamento e pré-processamento
    │   ├── model.py                       # Lógica de treinamento do modelo LightFM
    │   ├── modelo_sequencial.py           # Lógica de treinamento do modelo sequencial (SASRec)
    │   ├── preprocessing.py               # Funções de pré-processamento de dados
    │   └── train.py                       # Script que orquestra o treinamento dos modelos
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── itens/                             # Dados dos itens (usados no treinamento)
    │   ├── itens-parte1.csv
    │   ├── itens-parte2.csv
    │   └── itens-parte3.csv
    ├── treino/                            # Dados de treinamento
    │   ├── treino_parte1.csv
    │   ├── treino_parte2.csv
    │   ├── treino_parte3.csv
    │   ├── treino_parte4.csv
    │   ├── treino_parte5.csv
    │   └── treino_parte6.csv
    ├── .env
    ├── build_and_push.sh
    ├── DEPLOY.md                        # Instruções de build, push e deploy na AWS
    ├── README.md
    ├── requirements.txt
    └── requirements-api.txt

