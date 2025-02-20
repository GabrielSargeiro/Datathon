FROM python:3.11-slim

# Variáveis de ambiente para evitar geração de bytecode e garantir saída imediata
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Define o diretório de trabalho
WORKDIR /app

# Instala dependências do sistema necessárias para compilar extensões C
RUN apt-get update && apt-get upgrade -y && apt-get install -y gcc build-essential python3-dev libopenblas-dev liblapack-dev

# Atualiza pip e instala Cython, wheel e fixa setuptools (opcional)
RUN pip install --upgrade pip
RUN pip install cython wheel setuptools==65.5.0

# Copia e instala as dependências Python para a API
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copia o código de produção: a pasta app/ com a API e os artifacts
COPY ../app/ ./app/

# Expõe a porta da API
EXPOSE 5000

# Comando para iniciar a API com uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]