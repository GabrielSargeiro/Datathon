#!/bin/bash

# Cria diretório para artifacts se não existir
mkdir -p app/artifacts

# Muda para o diretório workflow onde está o docker-compose.yml
cd workflow

# Constrói a imagem
docker-compose build

# Executa o contêiner com limites de recursos
docker-compose run --rm lightfm-trainer
