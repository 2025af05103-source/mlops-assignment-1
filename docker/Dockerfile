FROM python:3.12-slim

WORKDIR /app

# Copy package files first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code and preprocessing module
COPY src/ src/
COPY models/ models/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
