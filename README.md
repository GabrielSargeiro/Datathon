### Requisitos:
- Ambiente virtual (venv) com Python 3.11.8.
- Instalar dependências:
    ```bash
    python -m pip install -r requirements.txt
    ```
- [Visual C++ Build Tools](https://visualstudio.microsoft.com/pt-br/visual-cpp-build-tools/) e ferramentas de build do Visual Studio 2022.
- Após instalar as dependências, execute:
  ```bash
  python -m spacy download pt_core_news_sm
  ```
     
### Passos para executar:

1. **Executar trainModel.py**, que vai gerar o model.pkl utilizando PLN e TF-IDF.
   
   ```
    workflow -> trainModel.py
   ```
   
2. **Executar build_lightfm.sh**, via Gitbash, que vai gerar o model_lightfm.pkl utilizando LightFM:

   ```bash
   ./build_lightfm.sh
   ```

## Deploy

Para instruções de build, push e deploy na AWS, consulte [DEPLOY.md](DEPLOY.md).

## Estrutura do Projeto

``` 
Datathon/
├── app/                       # API
│   ├── config.py              # Configurações da API
│   ├── main.py                # Ponto de entrada da API FastAPI
│   ├── artifacts/
│   │   ├── model.pkl          # Modelo TF-IDF/PLN
│   │   └── model_lightfm.pkl  # Modelo LightFM
│   ├── models/
│   │   └── data_models.py     # Schemas de dados (Pydantic)
│   └── routers/
│       └── recommendation.py  # Endpoints da API
├── workflow/                  
│   ├── trainModel.py          # Lógica de treinamento do modelo TF-IDF/PLN
│   ├── trainLightfm.py        # Pipeline leve para treinamento incremental do LightFM
│   ├── Dockerfile             # Dockerfile para o pipeline LightFM
│   └── docker-compose.yml     # Docker-compose para o pipeline LightFM
├── docker/                    # Docker para a API
│   ├── Dockerfile
│   └── docker-compose.yml
├── itens/                     # Dados dos itens (usados no treinamento)
│   ├── itens-parte1.csv
│   ├── itens-parte2.csv
│   └── itens-parte3.csv
├── treino/                    # Dados de treinamento
│   ├── treino_parte1.csv
│   ├── treino_parte2.csv
│   ├── treino_parte3.csv
│   ├── treino_parte4.csv
│   ├── treino_parte5.csv
│   └── treino_parte6.csv
├── .env
├── build_lightfm.sh
├── build_push_aws.sh
├── DEPLOY.md                  # Instruções de build, push e deploy na AWS
├── README.md
├── requirements.txt
└── requirements-api.txt
``` 
