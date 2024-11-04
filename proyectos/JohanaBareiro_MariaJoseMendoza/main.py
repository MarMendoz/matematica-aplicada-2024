# Importamos las librerías necesarias
import pandas as pd
import numpy as np
import re
import time
import nltk
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from fpdf import FPDF

# Descargar el lexicón de VADER solo la primera vez que se ejecute el código
nltk.download("vader_lexicon")

# ========================
# Módulo 1: Creación del Dataset
# ========================
# Aca se carga el dataset Sentiment140 y se hace un preprocesamiento básico del texto.

# Cargamos el archivo 'test_data.csv' 
data = pd.read_csv('test_data.csv')

# Función para limpiar el texto
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar caracteres especiales
    text = re.sub(r'\b\w\b', '', text)       # Eliminar palabras sueltas
    return text.strip().lower()              # Convertir todo a minúsculas y eliminar espacios

data['oracion_limpia'] = data['sentence'].apply(clean_text)

# ========================
# Módulo 2: Lexicón de Sentimientos
# ========================
# Usamos VADER de la librería NLTK para obtener puntajes positivos y negativos de cada oración.

sia = SentimentIntensityAnalyzer()
data['puntaje_positivo'] = data['oracion_limpia'].apply(lambda x: sia.polarity_scores(x)['pos'])
data['puntaje_negativo'] = data['oracion_limpia'].apply(lambda x: sia.polarity_scores(x)['neg'])

# ========================
# Módulo 3: Fuzzificación
# ========================
# Definimos las variables difusas de los puntajes de sentimiento y sus funciones de membresía.

positive = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'positive')
negative = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'negative')
sentiment = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'sentiment')

# Funciones de membresía
positive['bajo'] = fuzz.trimf(positive.universe, [0, 0, 0.5])
positive['medio'] = fuzz.trimf(positive.universe, [0, 0.5, 1])
positive['alto'] = fuzz.trimf(positive.universe, [0.5, 1, 1])

negative['bajo'] = fuzz.trimf(negative.universe, [0, 0, 0.5])
negative['medio'] = fuzz.trimf(negative.universe, [0, 0.5, 1])
negative['alto'] = fuzz.trimf(negative.universe, [0.5, 1, 1])

sentiment['negativo'] = fuzz.trimf(sentiment.universe, [-1, -1, 0])
sentiment['neutral'] = fuzz.trimf(sentiment.universe, [-0.5, 0, 0.5])
sentiment['positivo'] = fuzz.trimf(sentiment.universe, [0, 1, 1])

# ========================
# Módulo 4: Base de Reglas Difusas
# ========================
rule1 = ctrl.Rule(positive['alto'] & negative['bajo'], sentiment['positivo'])
rule2 = ctrl.Rule(positive['medio'] & negative['bajo'], sentiment['positivo'])
rule3 = ctrl.Rule(positive['bajo'] & negative['alto'], sentiment['negativo'])
rule4 = ctrl.Rule(positive['medio'] & negative['medio'], sentiment['neutral'])
rule5 = ctrl.Rule(positive['bajo'] & negative['bajo'], sentiment['neutral'])

sentiment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
sentiment_simulation = ctrl.ControlSystemSimulation(sentiment_ctrl)

# ========================
# Módulo 5: Defuzzificación
# ========================
fuzzy_results, sentiment_labels, fuzz_times, defuzz_times, total_times = [], [], [], [], []

# Aplicamos la fuzzificación y defuzzificación en cada tweet
for i, row in data.iterrows():
    # Fuzzificación
    start_fuzz = time.time()
    sentiment_simulation.input['positive'] = row['puntaje_positivo']
    sentiment_simulation.input['negative'] = row['puntaje_negativo']
    fuzz_time = time.time() - start_fuzz

    # Defuzzificación
    start_defuzz = time.time()
    sentiment_simulation.compute()
    defuzz_time = time.time() - start_defuzz
    total_time = fuzz_time + defuzz_time

    # Obtener el valor defuzzificado y resultado de la inferencia
    fuzzy_value = sentiment_simulation.output['sentiment']
    fuzzy_results.append(fuzzy_value)
    sentiment_label = 'Positiva' if fuzzy_value > 0.5 else ('Negativa' if fuzzy_value < -0.5 else 'Neutral')
    sentiment_labels.append(sentiment_label)

    # Guardar los tiempos y resultados
    fuzz_times.append(round(fuzz_time, 8))
    defuzz_times.append(round(defuzz_time, 8))
    total_times.append(round(total_time, 8))

# Guardamos los resultados en el DataFrame
data['sentimiento_fuzzificado'] = fuzzy_results
data['sentimiento_defuzzificado'] = sentiment_labels
data['tiempo_fuzzificacion'] = fuzz_times
data['tiempo_defuzzificacion'] = defuzz_times
data['tiempo_total'] = total_times

# ========================
# Módulo 6: Benchmarks y Exportación de Resultados
# ========================
summary = data['sentimiento_defuzzificado'].value_counts()
total_time_execution = data['tiempo_total'].sum()
average_time_execution = total_time_execution / len(data)

# Renombramos las columnas para quitar los acentos y agregar las columnas requeridas
data['label_original'] = data['sentimiento_defuzzificado']
data.rename(columns={
    'sentence': 'Oracion original',
    'label_original': 'label original',
    'puntaje_positivo': 'Puntaje Positivo',
    'puntaje_negativo': 'Puntaje Negativo',
    'sentimiento_fuzzificado': 'El resultado de la inferencia',
    'tiempo_fuzzificacion': 'Tiempo de fuzzificacion',
    'tiempo_defuzzificacion': 'Tiempo de desfuzzificacion',
    'tiempo_total': 'tiempo de ejecucion'
}, inplace=True)

# Reordenamos las columnas según se especificó
data[['Oracion original', 'label original', 'Puntaje Positivo', 'Puntaje Negativo',
      'El resultado de la inferencia', 'Tiempo de fuzzificacion', 'Tiempo de desfuzzificacion', 'tiempo de ejecucion']].to_csv('resultados_finales.csv', index=False)

print(f"Total de tweets procesados: {len(data)}")
print(f"Total positivos: {summary.get('Positiva', 0)}")
print(f"Total negativos: {summary.get('Negativa', 0)}")
print(f"Total neutrales: {summary.get('Neutral', 0)}")
print(f"Tiempo promedio de ejecución: {average_time_execution:.8f} s")

# ========================
# Creación del PDF con el análisis de resultados
# ========================

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Página de presentación
pdf.add_page()
pdf.set_font("Arial", size=16)
pdf.set_text_color(255, 105, 180)  # Rosa
pdf.cell(200, 10, txt="Analisis de Sentimientos con Logica Difusa", ln=True, align='C')
pdf.ln(10)
pdf.set_font("Arial", size=12)
pdf.set_text_color(148, 0, 211)  # Lila
pdf.multi_cell(0, 10, txt=(
    "Este informe presenta un analisis detallado del sentimiento de una serie de tweets "
    "utilizando tecnicas de logica difusa para clasificar el sentimiento en positivo, negativo o neutral."
))

# Resumen de clasificación
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Resumen de Clasificacion de Tweets", ln=True, align='C')
pdf.ln(10)
pdf.cell(200, 10, txt=f"Positivos: {summary.get('Positiva', 0)}", ln=True)
pdf.cell(200, 10, txt=f"Negativos: {summary.get('Negativa', 0)}", ln=True)
pdf.cell(200, 10, txt=f"Neutrales: {summary.get('Neutral', 0)}", ln=True)
pdf.cell(200, 10, txt=f"Total Procesado: {len(data)}", ln=True)
pdf.cell(200, 10, txt=f"Tiempo Total de Ejecucion (s): {total_time_execution:.8f}", ln=True)
pdf.cell(200, 10, txt=f"Tiempo Promedio de Ejecucion (s): {average_time_execution:.8f}", ln=True)

# Añadir análisis y gráficos de los primeros 10 tweets
for i in range(min(10, len(data))):
    row = data.iloc[i]
    pdf.add_page()
    plt.figure(figsize=(10, 6))

    # Graficar las funciones de membresía para el sentimiento positivo
    plt.plot(positive.universe, positive['bajo'].mf, 'r', linestyle='--', label='Positivo Bajo')
    plt.plot(positive.universe, positive['medio'].mf, 'm', linestyle='--', label='Positivo Medio')
    plt.plot(positive.universe, positive['alto'].mf, 'b', linestyle='--', label='Positivo Alto')
    plt.fill_between(positive.universe, positive['bajo'].mf, color='pink', alpha=0.3)
    plt.fill_between(positive.universe, positive['medio'].mf, color='violet', alpha=0.3)
    plt.fill_between(positive.universe, positive['alto'].mf, color='lightblue', alpha=0.3)

    # Agregar línea para el valor defuzzificado del sentimiento (COA)
    coa = row['El resultado de la inferencia']
    plt.axvline(x=coa, color='green', linestyle='-', linewidth=2, label='COA (Centroide)')

    plt.title(f'Fuzzy Membership Functions - Tweet {i+1}')
    plt.xlabel('Membership')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.savefig(f'temp_plot_{i}.png')
    plt.close()

    # Insertar la gráfica en el PDF
    pdf.image(f'temp_plot_{i}.png', x=10, y=30, w=180)
    pdf.ln(120)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, txt=(
        f"Tweet {i+1} Analisis Fuzzy:\n"
        f"- Sentimiento positivo: {row['Puntaje Positivo']:.6f}\n"
        f"- Sentimiento negativo: {row['Puntaje Negativo']:.6f}\n"
        f"- Sentimiento fuzzificado: {row['El resultado de la inferencia']:.6f}\n"
        f"- COA (Centroide del Area): {coa:.6f}\n"
        f"- Tiempo de fuzzificacion: {row['Tiempo de fuzzificacion']:.8f} s\n"
        f"- Tiempo de desfuzzificacion: {row['Tiempo de desfuzzificacion']:.8f} s\n"
        f"- Tiempo total: {row['tiempo de ejecucion']:.8f} s\n"
    ))

# Guardar el PDF
pdf.output('resultados.pdf')
