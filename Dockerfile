FROM python:3.11-slim

WORKDIR /app

# instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# instalar Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# instalar dependencias python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copiar app
COPY . .

RUN chmod +x start.sh

EXPOSE 8501
EXPOSE 11434

CMD ["./start.sh"]