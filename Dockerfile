# Etapa 1: Imagem base com suporte a GPU
FROM tensorflow/tensorflow:latest-gpu

# Etapa 2: Configurar variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Etapa 3: Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Etapa 4: Copiar arquivos do projeto
WORKDIR /app
COPY requirements.txt .
COPY preprocessing.py .
COPY treinamento_modelo_sasrec.py .

# Etapa 5: Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Etapa 6: Definir comando padrão
CMD ["python", "treinamento_modelo_sasrec.py"]