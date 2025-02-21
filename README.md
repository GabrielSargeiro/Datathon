 ## Treino
 
 Para treinar o modelo, execute:
 
 ```bash
 python -m pip install -r requirements.txt
 ```
 
 ### Requisitos:
 - Ambiente virtual (venv) com Python 3.11.8.
 - [Visual C++ Build Tools](https://visualstudio.microsoft.com/pt-br/visual-cpp-build-tools/) e ferramentas de build do Visual Studio 2022.
 - Processamento de Linguagem Natural (PLN):
   - Após instalar as dependências, execute:
     ```bash
     python -m spacy download pt_core_news_sm
     ```
 - O pipeline de treinamento utiliza técnicas de PLN e TF-IDF para processar os dados dos itens e gerar um modelo.
 - O modelo treinado é salvo em `app/artifacts/model.pkl`.
 
 ## Deploy
 
 Para instruções de build, push e deploy na AWS, consulte [DEPLOY.md](DEPLOY.md).
 
 No deploy, a API carrega o modelo salvo e expõe endpoints via FastAPI para fornecer recomendações.
 
 ## Estrutura do Projeto
 
 ```
 Datathon/
 ├── app/
 │   ├── config.py                      # Configurações da API
 │   ├── main.py                        # Ponto de entrada da API FastAPI
 │   ├── artifacts/
 │   │   └── model.pkl                  # Modelo treinado para inferência
 │   ├── models/
 │   │   └── data_models.py             # Schemas de dados (Pydantic)
 │   └── routers/
 │       └── recommendation.py          # Endpoints da API
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
 ├── train.py                           # Lógica de treinamento do modelo TF-IDF/PLN
 ├── build_and_push.sh
 ├── DEPLOY.md                         # Instruções de build, push e deploy na AWS
 ├── README.md
 ├── requirements.txt
 └── requirements-api.txt
 ```
