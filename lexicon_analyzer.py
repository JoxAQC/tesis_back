# backend/lexicon_analyzer.py
import pandas as pd
import re
import unicodedata
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import os
import base64
from io import BytesIO

# Preparación inicial
try:
    nlp = spacy.load("es_core_news_sm")
    stopwords_es = set(stopwords.words('spanish'))
except:
    # Si no están instalados, intenta descargarlos
    import spacy.cli
    import nltk
    spacy.cli.download("es_core_news_sm")
    nltk.download('stopwords')
    nlp = spacy.load("es_core_news_sm")
    stopwords_es = set(stopwords.words('spanish'))

tqdm.pandas()
modificadores = {'no', 'nunca', 'jamás', 'nada', 'nadie', 'ninguno', 'ni', 'poco', 'muy', 'tan', 'bastante', 'demasiado', 'otro', 'mucho'}
verbos_auxiliares = {'ser', 'estar', 'haber'}

def _limpiar_texto(texto):
    texto = ''.join(c for c in unicodedata.normalize('NFD', str(texto).lower()) if unicodedata.category(c) != 'Mn')
    doc = nlp(texto)
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and (token.lemma_ not in stopwords_es or token.lemma_ in modificadores)
        and token.lemma_ not in verbos_auxiliares
    ]
    return ' '.join(tokens)

def _analyze_sentiment_advanced(text, lexicon_entries):
    doc = nlp(text)
    words = [token.text for token in doc if not token.is_punct]
    total_words = len(words)
    if total_words == 0:
        return pd.Series([0, 0, 0])
    
    anger_score, joy_score, sadness_score = 0, 0, 0
    for entry in lexicon_entries:
        phrase = entry['Text'].lower()
        regex_pattern = r'\b' + re.escape(phrase) + r'\b'
        matches = len(re.findall(regex_pattern, text.lower()))
        anger_score += matches * entry.get('Anger', 0)
        joy_score += matches * entry.get('Joy', 0)
        sadness_score += matches * entry.get('Sadness', 0)
    
    anger_norm = (anger_score / total_words) * 100
    joy_norm = (joy_score / total_words) * 100
    sadness_norm = (sadness_score / total_words) * 100
    
    return pd.Series([anger_norm, joy_norm, sadness_norm])

def analyze_with_lexicon(reviews_df):
    """
    Aplica el análisis de sentimiento con léxico a un DataFrame de reseñas.
    Args:
        reviews_df (pd.DataFrame): DataFrame con la columna 'review'.
    Returns:
        pd.DataFrame: DataFrame con las columnas de sentimiento y predicción.
    """
    if not isinstance(reviews_df, pd.DataFrame):
        reviews_df = pd.read_excel(reviews_df)

    # Cargar léxico
    df_lexicon = pd.read_excel("Lexicon.xlsx", sheet_name="lexiconv2_lematizado")
    df_lexicon['Text'] = df_lexicon['Text'].astype(str).str.lower()
    lexicon_entries = df_lexicon.to_dict(orient='records')
    
    reviews_df['reseña_limpia'] = reviews_df['review'].progress_apply(_limpiar_texto)
    reviews_df[['anger', 'joy', 'sadness']] = reviews_df['reseña_limpia'].progress_apply(
        lambda x: _analyze_sentiment_advanced(x, lexicon_entries)
    )

    def clasificar_por_emociones(row):
        joy = row['joy']
        neg = row['anger'] + row['sadness']
        if joy > neg:
            return 'positivo'
        elif neg > joy:
            return 'negativo'
        else:
            return 'neutro'

    reviews_df['prediccion_lexicon'] = reviews_df.apply(clasificar_por_emociones, axis=1)
    
    return reviews_df[['review', 'reseña_limpia', 'anger', 'joy', 'sadness', 'prediccion_lexicon']]

def generate_lexicon_plots(df_reviews):
    """
    Genera y devuelve los gráficos de análisis del léxico en formato Base64.
    Args:
        df_reviews (pd.DataFrame): DataFrame con resultados del análisis.
    Returns:
        dict: Diccionario con nombres de gráficos y sus datos en Base64.
    """
    plots = {}
    
    # 1. Gráfico de conteo por polaridad
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_reviews, x='prediccion_lexicon', hue='prediccion_lexicon', palette='Set2', legend=False)
    plt.title("Distribución de Polaridades (Lexicon)")
    plt.xlabel("Polaridad")
    plt.ylabel("Cantidad de Comentarios")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['polaridad_count'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # 2. Promedio de emociones por polaridad
    plt.figure(figsize=(10, 6))
    emociones_promedio = df_reviews.groupby('prediccion_lexicon')[['anger', 'joy', 'sadness']].mean().reset_index()
    emociones_promedio = emociones_promedio.melt(id_vars='prediccion_lexicon', var_name='emocion', value_name='promedio')
    sns.barplot(data=emociones_promedio, x='prediccion_lexicon', y='promedio', hue='emocion', palette={'anger': 'tomato', 'joy': 'gold', 'sadness': 'skyblue'})
    plt.title("Promedio de Intensidad Emocional por Polaridad (Lexicon)")
    plt.xlabel("Polaridad")
    plt.ylabel("Intensidad Promedio")
    plt.legend(title="Emociones")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['promedio_emociones'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # 3. Nube de palabras (general)
    texto_completo = " ".join(review for review in df_reviews['reseña_limpia'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_completo)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Nube de Palabras General (Lexicon)")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['nube_palabras'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return plots