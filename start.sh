#!/bin/bash
set -e

# Arrancar servidor de Ollama en background
ollama serve &

# Esperar a que Ollama responda
sleep 5

# Descargar el modelo si no está
ollama pull llama3.2:3b

# Lanzar Streamlit
streamlit run app-LLM.py --server.port=8501 --server.address=0.0.0.0