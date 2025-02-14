# docker/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copia apenas o arquivo de dependências e instala as bibliotecas necessárias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia somente os arquivos essenciais para a API (apenas o model e o app)
COPY app.py .
COPY model.pkl .

EXPOSE 5000

CMD ["python", "app.py"]
