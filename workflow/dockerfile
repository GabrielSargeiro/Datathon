FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc \
      python3-dev \
      libopenblas-dev \
      libomp-dev && \
    rm -rf /var/lib/apt/lists/*

# Copia os arquivos necessários
COPY ../requirements.txt .
COPY ../workflow/trainLightfm.py .
COPY ../app/artifacts/model.pkl artifacts/model.pkl

# Instala dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Garante permissões corretas
RUN chmod +x /app/trainLightfm.py

CMD ["python", "trainLightfm.py"]
