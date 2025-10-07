# 💼 Portafolio Profesional

## 👋 Sobre mí

<img src="https://github.com/barcklan.png" width="120" align="left" style="border-radius: 50%; margin-right: 20px;">

Soy un profesional apasionado por el **análisis de datos**, la **modelación estadística** y el **desarrollo de soluciones basadas en Machine Learning** que contribuyan a la toma de decisiones informadas.

**Nombre:** Claudio Andrés Díaz Vargas  
**Especialización:** Ingeniero en Estadística, especializado en Machine Learning y Ciencia de Datos  
**Correo:** [cdiazv.ies@gmail.com](mailto:cdiazv.ies@gmail.com)  
**GitHub:** [https://github.com/barcklan](https://github.com/barcklan)

Tengo experiencia en **análisis exploratorio de datos**, **modelado estadístico**, **técnicas de aprendizaje supervisado y no supervisado**, **optimización de modelos**, **visualización de datos** y **automatización de procesos analíticos** con Python.

Mi objetivo es **aplicar mis conocimientos en estadística y Machine Learning** para diseñar soluciones basadas en datos que apoyen la toma de decisiones y generen impacto positivo.

---

## 🚀 Proyectos Destacados

A continuación, se presentan tres de mis proyectos más relevantes, donde aplico conocimientos en análisis, programación y documentación técnica.

---

## 🧠 PROYECTO 1 — Sistema Inteligente de *Scoring* Crediticio con Redes Neuronales Profundas (DNN)

### 🎯 Objetivo

Diseñar, entrenar y evaluar un modelo de red neuronal profunda para predecir la probabilidad de impago de clientes bancarios, utilizando un conjunto de datos realista.  
El modelo debe ser explicable, eficiente y presentar resultados interpretables para su uso en contextos financieros.

### 🧩 Contexto

Las entidades financieras deben decidir si otorgan o no un crédito a un cliente.  
Esta decisión, si se basa en modelos poco explicables, puede generar sesgos, exclusiones injustas o pérdidas económicas.  

Se busca construir un modelo moderno, basado en **redes neuronales profundas**, que sea a la vez **preciso y explicable**, permitiendo a las instituciones mejorar la calidad de sus decisiones crediticias.

### 📊 Resumen

Se desarrolló un sistema de **scoring crediticio** basado en **DNN** para predecir la probabilidad de impago de clientes bancarios.

Tras un **análisis exploratorio** que identificó diferencias entre buenos y malos pagadores (en **monto y duración del crédito**), se aplicó **SMOTE** para balancear clases y se entrenaron dos modelos:

- **DNN simple:** *accuracy* de **70,5%**, **AUC = 0,78**, mostrando buena generalización y equilibrio entre clases.  
- **ResNet tabular:** *accuracy* de **64%**, **AUC = 0,64**, mejor detección de clientes *bad* pero más falsos positivos.

📈 La **DNN simple** se posiciona como la opción más confiable y efectiva, aunque puede mejorarse la predicción de clientes solventes y la interpretabilidad.

### 🧰 Tecnologías Utilizadas

- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **SMOTE (imbalanced-learn)**

🔗 [Ver proyecto completo](./proyecto1)

---

## 🧬 PROYECTO 2 — Clasificación de Notas Clínicas para Detección Temprana de Afecciones  
### Con enfoque ético y mitigación de sesgos

### 🎯 Objetivo

Desarrollar un sistema de **procesamiento de lenguaje natural (NLP)** capaz de clasificar textos médicos (notas clínicas, síntomas, diagnósticos) según su **gravedad clínica (leve, moderado, severo)**.  
El modelo debe incluir **buenas prácticas de preprocesamiento, evaluación y mitigación de sesgos lingüísticos y sociales**.

### 🧩 Contexto

Los registros médicos en texto libre contienen información valiosa para detectar la gravedad de una afección de forma temprana.  
Sin embargo, su interpretación manual requiere tiempo, experiencia y puede verse afectada por **sesgos humanos**.

El sistema desarrollado clasifica automáticamente las notas clínicas según su **nivel de gravedad**, ayudando a profesionales de la salud a **priorizar pacientes** y mejorar la eficiencia hospitalaria.  
Además, se realizó un análisis ético sobre los posibles sesgos y se aplicaron métodos de **interpretabilidad** para garantizar confianza en su aplicación.

### 📊 Resumen

Se compararon dos enfoques:

- **Naive Bayes con TF-IDF:** enfoque clásico, eficiente y explicable.  
- **BERT en español:** modelo contextualizado de última generación.

Ambos lograron **métricas perfectas en validación**, lo que evidenció tanto la capacidad de separación de los datos como un **riesgo de sobreajuste**.  
Se aplicaron técnicas de **LIME** para interpretabilidad y se evaluaron riesgos éticos y sesgos potenciales.

📈 El proyecto demuestra cómo los modelos de NLP pueden aplicarse en contextos clínicos de forma **efectiva, transparente y responsable**.

### 🧰 Tecnologías Utilizadas

- **Python**
- **Scikit-learn**
- **Transformers (Hugging Face)**
- **BERT Multilingual / BETO**
- **LIME**
- **NLTK / spaCy**
- **Pandas / NumPy**

🔗 [Ver proyecto completo](./proyecto2)

---

## 🧩 PROYECTO 3 — Interpretabilidad de Modelos Predictivos usando LIME y SHAP

### 🎯 Objetivo

Aplicar herramientas de **explicabilidad de modelos**, como **LIME** y **SHAP**, para analizar y justificar el comportamiento de un modelo de clasificación, destacando la importancia de la transparencia en la inteligencia artificial.

### 🧩 Contexto

Formar parte de un equipo que usa **IA en decisiones críticas** implica construir modelos **explicables y éticamente responsables**.  
Este proyecto explora cómo la interpretabilidad permite **auditar la lógica interna** de los modelos y detectar errores o sesgos ocultos.

### 📊 Resumen

Se analizó un modelo de **Random Forest** para predecir **enfermedades cardíacas**, utilizando **LIME** y **SHAP**.  
El modelo alcanzó **88.6% de precisión**, pero las explicaciones revelaron **fallas críticas**:

- Uso correcto de variables relevantes (**ECG**, **angina**, **frecuencia cardíaca máxima**).  
- **Manejo incorrecto del colesterol**: interpretó valores bajos como riesgosos y altos como protectores, debido a datos con valores `0.0`.

📉 Sin interpretabilidad, este error habría pasado inadvertido, comprometiendo decisiones clínicas.  
El caso demuestra que **precisión sin transparencia** no es suficiente: la interpretabilidad garantiza **auditoría, confianza y ética** en modelos de IA.

### 🧰 Tecnologías Utilizadas

- **Python**
- **Scikit-learn**
- **Random Forest**
- **LIME**
- **SHAP**
- **Matplotlib / Seaborn**
- **Pandas / NumPy**

🔗 [Ver proyecto completo](./proyecto3)

---

## 🧭 Organización y Buenas Prácticas

Este portafolio está organizado de manera clara y estructurada:

- Navegación sencilla entre secciones.  
- Documentación técnica y reflexiva.  
- Redacción cuidada, ortografía revisada y estilo profesional.

---

## 🌐 Enlace al Portafolio

Puedes acceder a este portafolio directamente en GitHub:  
🔗 [https://github.com/barcklan/portfolio-profesional](https://github.com/barcklan/portfolio-profesional)
