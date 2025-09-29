# backend/scraper.py
import re
import time
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from dateutil.relativedelta import relativedelta
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def _convertir_fecha(fecha_texto):
    """Convierte fechas relativas a formato mm/yy."""
    hoy = datetime.today()
    texto_a_numeral = {"un": 1, "una": 1}
    match = re.search(r"(?:Hace|)(?:\s+|)(un|una|\d+)\s+(día|semana|mes|año)s?\s*(atrás)?", fecha_texto, re.IGNORECASE)
    
    if match:
        cantidad_texto = match.group(1).lower()
        unidad = match.group(2).lower()
        cantidad = texto_a_numeral.get(cantidad_texto, int(cantidad_texto) if cantidad_texto.isdigit() else 1)
        
        if "día" in unidad:
            fecha_final = hoy - timedelta(days=cantidad)
        elif "semana" in unidad:
            fecha_final = hoy - timedelta(weeks=cantidad)
        elif "mes" in unidad:
            fecha_final = hoy - relativedelta(months=cantidad)
        elif "año" in unidad:
            fecha_final = hoy - relativedelta(years=cantidad)
        
        return fecha_final.strftime("%m/%y")
    return fecha_texto

def get_google_maps_reviews(place_url):
    """
    Realiza web scraping en una URL de Google Maps y devuelve un DataFrame con las reseñas.
    """
    service = Service()
    options = webdriver.ChromeOptions()
    # Agregar opción para ejecutar sin interfaz gráfica (modo headless)
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(place_url)
    reviews_data = []

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "DUwDvf")))
        place_name_element = driver.find_element(By.CLASS_NAME, "DUwDvf")
        place_name_text = place_name_element.text.strip()
        
        # Intenta hacer clic en la pestaña de reseñas
        reviews_tab = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Reseñas')]"))
        )
        reviews_tab.click()
        
        scrollable_div = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde"))
        )

        last_review_count = 0
        scroll_attempts = 0
        max_attempts = 8
        while scroll_attempts < max_attempts:
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
            time.sleep(3)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            reviews_elements = soup.find_all("div", class_="jftiEf")
            current_review_count = len(reviews_elements)
            if current_review_count == last_review_count:
                scroll_attempts += 1
            else:
                last_review_count = current_review_count
                scroll_attempts = 0
            
            # Expande todas las reseñas
            more_buttons = driver.find_elements(By.CLASS_NAME, "w8nwRe")
            for button in more_buttons:
                try:
                    driver.execute_script("arguments[0].click();", button)
                except:
                    pass

        soup = BeautifulSoup(driver.page_source, "html.parser")
        reviews_elements = soup.find_all("div", class_="jftiEf")

        for review_element in reviews_elements:
            try:
                review_text_element = review_element.find("span", class_="wiI7pd")
                review_text = review_text_element.text.strip() if review_text_element else "N/A"
                raw_rating = review_element.find("span", class_="kvMYJc")["aria-label"]
                rating = float(re.search(r"(\d+)", raw_rating).group(1))
                review_date_element = review_element.find("span", class_="rsqaWe")
                review_date = _convertir_fecha(review_date_element.text.strip()) if review_date_element else "N/A"
                reviews_data.append({
                    "name": place_name_text,
                    "review": review_text,
                    "estrellas": rating,
                    "review_date": review_date
                })
            except Exception as e:
                print(f"Error procesando una reseña: {e}")
    except Exception as e:
        print(f"Error general en el scraper: {e}")
    finally:
        driver.quit()

    return pd.DataFrame(reviews_data)