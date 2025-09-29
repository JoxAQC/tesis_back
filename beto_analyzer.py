# backend/beto_analyzer.py
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import base64
from io import BytesIO

# Configuración inicial
tqdm.pandas()
analyzer = pipeline(
    "text-classification",
    model="finiteautomata/beto-sentiment-analysis",
    tokenizer="finiteautomata/beto-sentiment-analysis"
)

def analyze_with_beto(reviews_df):
    """
    Aplica el análisis de sentimiento con el modelo BETO a un DataFrame de reseñas.
    Args:
        reviews_df (pd.DataFrame): DataFrame con la columna 'review'.
    Returns:
        pd.DataFrame: DataFrame con la predicción de BETO.
    """
    if not isinstance(reviews_df, pd.DataFrame):
        reviews_df = pd.read_excel(reviews_df)
    
    def _clasificar_con_beto(texto):
        try:
            # Limitar a 512 tokens para evitar errores
            result = analyzer(str(texto)[:512])
            return result[0]['label'].lower()
        except:
            return 'neutro' # Manejo de errores para texto vacío o problemático

    reviews_df['review_clean'] = reviews_df['review'].astype(str).str.strip()
    print("🔍 Analizando reseñas con BETO...")
    reviews_df['prediccion_beto'] = reviews_df['review_clean'].progress_apply(_clasificar_con_beto)
    
    # Mapeo de etiquetas
    label_map = {
        'pos': 'positivo',
        'neg': 'negativo',
        'neu': 'neutro'
    }
    reviews_df['prediccion_beto'] = reviews_df['prediccion_beto'].map(label_map).fillna(reviews_df['prediccion_beto'])
    
    return reviews_df[['review', 'review_clean', 'prediccion_beto']]

def generate_beto_plots(df_reviews):
    """
    Genera y devuelve los gráficos de análisis de BETO en formato Base64.
    Args:
        df_reviews (pd.DataFrame): DataFrame con resultados del análisis.
    Returns:
        dict: Diccionario con nombres de gráficos y sus datos en Base64.
    """
    plots = {}
    
    # 1. Gráfico de conteo por polaridad
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_reviews, x='prediccion_beto', hue='prediccion_beto', palette='Set3', legend=False)
    plt.title("Distribución de Polaridades (BETO)")
    plt.xlabel("Polaridad")
    plt.ylabel("Cantidad de Comentarios")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['polaridad_count_beto'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # 2. Nube de palabras (general)
    texto_completo = " ".join(review for review in df_reviews['review_clean'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto_completo)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Nube de Palabras General (BETO)")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plots['nube_palabras_beto'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return plots