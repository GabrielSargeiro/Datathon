FROM python:3.12-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo de requisitos e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia os arquivos do projeto para o container
COPY requirements.txt .
COPY app/ ./app/
COPY model.pkl .

# Exponha a porta 5000, que é a porta que sua API irá utilizar
EXPOSE 5000

# Inicia a API com Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]