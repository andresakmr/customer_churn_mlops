FROM python:3.12-slim

WORKDIR /code

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir keras==3.13.2 tensorflow==2.18.0 numpy==1.26.4 fastapi uvicorn mlflow dagshub

COPY main.py .
COPY scaler.pkl .
COPY . /code

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]