# Ejecución del Proyecto de Análisis de Sentimiento Difuso en Tweets

Este proyecto realiza un análisis de sentimiento en tweets utilizando técnicas de **fuzzificación** y **defuzzificación**, generando un puntaje continuo para clasificar el sentimiento como positivo, negativo o neutral. Utiliza el lexicón `VADER` de la librería NLTK para calcular los puntajes de sentimiento y la biblioteca `scikit-fuzzy` para el análisis difuso. A continuación, se detalla cómo preparar el entorno de trabajo, ejecutar el análisis y entender los resultados generados.

> **Nota**: El proyecto descargará automáticamente el lexicón de VADER la primera vez que se ejecute, asegurando que solo sea necesario una única descarga inicial.

## Archivos Incluidos

- **main.py**: Script principal que ejecuta el análisis de sentimiento difuso.
- **test_data.csv**: Dataset que contiene los tweets a analizar. Incluye la columna `sentence` (texto del tweet).
- **requirements.txt**: Lista de todas las dependencias necesarias para el proyecto.
- **README.md**: Guía completa para configurar y ejecutar el proyecto.

## Preparación del Entorno de Trabajo

Para aislar las dependencias del proyecto y mantener un entorno limpio, es recomendable crear un **entorno virtual**. Esto asegura que todos los paquetes se instalen solo en el contexto de este proyecto.

### Crear un Entorno Virtual

1. Abre la terminal en la carpeta del proyecto.
2. Crea el entorno virtual con el siguiente comando:
   ```bash
   python -m venv .venv
   ```

### Activar el Entorno Virtual

- En **Windows**:
  ```bash
  .venv\Scripts\activate
  ```

- En **macOS y Linux**:
  ```bash
  source .venv/bin/activate
  ```

### Instalar las Dependencias

Con el entorno virtual activado, instala las dependencias usando:

```bash
pip install -r requirements.txt
```

Esto instalará las bibliotecas necesarias, incluidas `nltk`, `pandas`, `scikit-fuzzy`, `matplotlib`, y `fpdf`, entre otras.

## Ejecución del Proyecto

### Paso 1: Confirmar Archivos Necesarios
Asegúrate de que los archivos `main.py`, `test_data.csv` y las dependencias esten configuradas correctamente. El lexicón de VADER se descargará automáticamente al ejecutar el script principal si aún no está disponible.

### Paso 2: Ejecutar el Script Principal
Con el entorno virtual activado, ejecuta el script principal en la terminal:

```bash
python main.py
```

Este script realiza el análisis completo, que incluye los siguientes pasos:

1. **Carga y Preprocesamiento del Dataset**: Se carga el dataset `test_data.csv` y se limpia el texto de los tweets. Se eliminan caracteres especiales y se convierte todo el texto a minúsculas para un análisis más homogéneo.
2. **Análisis de Sentimientos con VADER**: Se utiliza el lexicón de VADER para calcular los puntajes de sentimiento positivo y negativo de cada tweet, agregando los resultados al dataset.
3. **Fuzzificación de Puntajes**: Los puntajes de sentimiento se convierten en valores difusos utilizando funciones de membresía triangulares, con categorías como `bajo`, `medio`, y `alto`.
4. **Aplicación de Reglas Difusas**: Se aplican reglas difusas (IF-THEN) para determinar el sentimiento general del tweet.
5. **Defuzzificación**: Los resultados difusos se transforman en un valor preciso usando el método del centroide, obteniendo un puntaje continuo que clasifica el sentimiento como positivo, negativo o neutral.
6. **Exportación de Resultados**: Los resultados se guardan en un archivo CSV y se genera un informe en PDF.

### Paso 3: Revisar los Resultados
Al finalizar la ejecución, el script generará dos archivos en el mismo directorio:

- **resultados_finales.csv**: Este archivo contiene el análisis de sentimiento completo para cada tweet, con las siguientes columnas:
  - `Oracion original`: El texto original del tweet.
  - `label original`: La etiqueta original del sentimiento en el dataset (si estuviera disponible).
  - `Puntaje Positivo`: Puntaje de sentimiento positivo calculado con VADER.
  - `Puntaje Negativo`: Puntaje de sentimiento negativo calculado con VADER.
  - `El resultado de la inferencia`: Puntaje difuso de sentimiento tras el proceso de defuzzificación.
  - `Tiempo de fuzzificacion`: Tiempo empleado en la fuzzificación del tweet.
  - `Tiempo de desfuzzificacion`: Tiempo empleado en la defuzzificación del tweet.
  - `tiempo de ejecucion`: Tiempo total de procesamiento de cada tweet.

- **resultados.pdf**: Informe en formato PDF que incluye un resumen de la clasificación de los tweets (positivos, negativos, neutrales) y gráficos de los análisis difusos realizados en algunos de los tweets. Este PDF es ideal para presentar los resultados de una manera visual y resumida. Si se quiere volver a ejecutar hay que eliminar las imagenes, el pdf, y el csv resultado que genera la primera ejecución. Luego se vuelve a ejecutar sin problemas.

## Notas Adicionales

- **Primera Ejecución**: La primera vez que se ejecuta el proyecto, se descarga automáticamente el lexicón de VADER desde NLTK. Esta descarga solo es necesaria una vez.
- **Revisión Visual de Resultados**: En el PDF generado, se incluyen gráficos que muestran las **funciones de membresía** para algunos tweets seleccionados, con la indicación del valor final difuso (`Centroide del Área`), lo cual ayuda a comprender mejor cómo se realizó la clasificación.

## Conclusión

El archivo `resultados_finales.csv` incluye todas las métricas de sentimiento difuso calculadas para cada tweet, lo cual es ideal para analizar los resultados con mayor detalle. Además, el informe PDF proporciona un resumen visual útil para comprender cómo se llevó a cabo el análisis de sentimientos.

Este enfoque permite un **análisis detallado y flexible** de los sentimientos en tweets, utilizando tanto técnicas de procesamiento de lenguaje natural como de lógica difusa, lo cual lo hace particularmente efectivo para manejar la **incertidumbre y ambigüedad** del lenguaje humano.

