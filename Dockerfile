FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY coderag ./coderag
COPY frontend ./frontend

EXPOSE 8090

# Bind 0.0.0.0 inside the container; compose maps the port to host loopback only.
CMD ["uvicorn", "coderag.main:app", "--host", "0.0.0.0", "--port", "8090"]
