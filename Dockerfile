FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY coderag ./coderag
COPY frontend ./frontend

EXPOSE 8090

CMD ["uvicorn", "coderag.main:app", "--host", "127.0.0.1", "--port", "8090"]
