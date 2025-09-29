#!/bin/bash

# 1. Instalar todas las librerías de Python
pip install -r requirements.txt

# 2. Descargar el modelo de SpaCy (asumiendo que usas el pequeño de español)
# ¡IMPORTANTE! El modelo se descarga DEBE ser el que llamas en tu código.
python -m spacy download es_core_news_sm

# 3. Descargar el recurso 'stopwords' de NLTK
python -m nltk.downloader stopwords