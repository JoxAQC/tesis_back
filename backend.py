# backend/backend.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import pandas as pd
from scraper import get_google_maps_reviews
from lexicon_analyzer import analyze_with_lexicon, generate_lexicon_plots
from beto_analyzer import analyze_with_beto, generate_beto_plots

app = FastAPI(
    title="Análisis de Sentimientos de Aseguradoras",
    description="API para procesar reseñas de Google Maps y analizarlas con Lexicón y BETO."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScrapeRequest(BaseModel):
    url: str

@app.post("/analyze")
async def analyze_reviews(request: ScrapeRequest):
    """
    Endpoint para procesar el URL de Google Maps y realizar el análisis de sentimientos.
    """
    url = request.url
    print(f"Recibida petición para analizar URL: {url}")

    # 1. Web Scraping
    try:
        reviews_df = get_google_maps_reviews(url)
        if reviews_df.empty:
            raise HTTPException(status_code=404, detail="No se encontraron reseñas o la URL es incorrecta.")
        print("✅ Scraping completado. Reseñas obtenidas.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el Web Scraping: {e}")

    # 2. Análisis con Lexicón
    try:
        lexicon_results = analyze_with_lexicon(reviews_df.copy())
        lexicon_plots_data = generate_lexicon_plots(lexicon_results)
        print("✅ Análisis con Lexicón completado.")
    except Exception as e:
        print(f"Error en el análisis con Lexicón: {e}")
        lexicon_results = pd.DataFrame()
        lexicon_plots_data = {}

    # 3. Análisis con BETO
    try:
        beto_results = analyze_with_beto(reviews_df.copy())
        beto_plots_data = generate_beto_plots(beto_results)
        print("✅ Análisis con BETO completado.")
    except Exception as e:
        print(f"Error en el análisis con BETO: {e}")
        beto_results = pd.DataFrame()
        beto_plots_data = {}

    # Devolver los resultados
    return {
        "status": "success",
        "lexicon_analysis": lexicon_results.to_dict(orient="records"),
        "lexicon_plots": lexicon_plots_data,
        "beto_analysis": beto_results.to_dict(orient="records"),
        "beto_plots": beto_plots_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
