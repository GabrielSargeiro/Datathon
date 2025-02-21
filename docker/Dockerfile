FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y gcc build-essential python3-dev libopenblas-dev liblapack-dev

RUN pip install --upgrade pip
RUN pip install cython wheel setuptools==65.5.0

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY ../app/ ./app/

COPY ../app/artifacts/ ./artifacts/

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
