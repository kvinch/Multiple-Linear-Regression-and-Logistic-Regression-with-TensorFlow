# Student Performance & Dropout Risk Analysis 

Este proyecto implementa modelos de **Regresión Lineal Múltiple** y **Regresión Logística** utilizando **TensorFlow** y **Python**. El objetivo es analizar los hábitos de estudio y predecir tanto la calificación de los exámenes (`exam_score`) como el riesgo de deserción escolar (`dropout_risk`).

## Características

* **Regresión Lineal (TensorFlow):** Predicción de puntajes basada en factores socio-económicos y hábitos de estudio.
* **Regresión Logística (TensorFlow):** Clasificación binaria para determinar el riesgo de deserción.
* **Análisis Estadístico:** Pruebas de normalidad, homocedasticidad (Breusch-Pagan) y multicolinealidad (VIF).
* **Preprocesamiento Automático:** Manejo de variables categóricas mediante OneHotEncoding y escalado de datos.

## 🛠️ Instalación

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/kvinch/Multiple-Linear-Regression-and-Logistic-Regression-with-TensorFlow]
2. Instala las dependencias
   ```bash
   pip install -r requirements.txt
3. Ejecuta el script principal
   ```bash
   python main_script.py
