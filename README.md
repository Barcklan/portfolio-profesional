# 💼 Portafolio Profesional

<table>
<tr>
<td width="150" align="center">

<img src="https://github.com/barcklan.png" width="120" style="border-radius: 50%; margin-right: 20px;">

</td>
<td>

## 👋 Sobre mí

Soy un profesional apasionado por el **análisis de datos**, la **modelación estadística** y el **desarrollo de soluciones basadas en Machine Learning** que contribuyan a la toma de decisiones informadas.

**Nombre:** Claudio Andrés Díaz Vargas  
**Especialización:** Ingeniero en Estadística, especializado en Machine Learning y Ciencia de Datos  
**Correo:** [cdiazv.ies@gmail.com](mailto:cdiazv.ies@gmail.com)  
**GitHub:** [https://github.com/barcklan](https://github.com/barcklan)

Tengo experiencia en **análisis exploratorio de datos**, **modelado estadístico**, **técnicas de aprendizaje supervisado y no supervisado**, **optimización de modelos**, **visualización de datos** y **automatización de procesos analíticos** con Python.

Mi objetivo es **aplicar mis conocimientos en estadística y Machine Learning** para diseñar soluciones basadas en datos que apoyen la toma de decisiones y generen impacto positivo.

</td>
</tr>
</table>

---

## 🚀 Proyectos Destacados

A continuación, se presentan tres de mis proyectos más relevantes, donde aplico conocimientos en análisis, programación y documentación técnica.

---

## 🧠 Proyecto 1 — Sistema Inteligente de Scoring Crediticio (DNN)

> **Evaluación Modular - Módulo 7**  
> **Tema:** Redes Neuronales Profundas aplicadas a riesgo crediticio.

---

### 🎯 Objetivo
Diseñar, entrenar y evaluar un modelo de red neuronal profunda para predecir la probabilidad de impago de clientes bancarios, utilizando un conjunto de datos realista.  
El modelo debe ser explicable, eficiente y presentar resultados interpretables para su uso en contextos financieros.

---

### 🧩 Contexto
Las entidades financieras deben decidir si otorgan o no un crédito a un cliente. Esta decisión, si se basa en modelos poco explicables, puede generar sesgos, exclusiones injustas o pérdidas económicas.  
Este proyecto busca construir un modelo **moderno, preciso y explicable**, basado en **redes neuronales profundas (DNN)**, que permita mejorar las decisiones crediticias.

---

### 📊 Resumen
Se desarrolló un sistema de **scoring crediticio** basado en **redes neuronales profundas (DNN)** para predecir la probabilidad de impago.  
Tras aplicar **SMOTE** para balancear las clases, se entrenaron dos modelos:

- 🧠 **DNN simple** → Accuracy: **70.5%**, AUC: **0.78**  
- ⚙️ **ResNet tabular** → Accuracy: **64%**, AUC: **0.64**

📈 La **DNN simple** mostró mejor generalización, equilibrio y estabilidad, siendo la opción más confiable para decisiones crediticias.

---

### 🧰 Tecnologías Utilizadas
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- SMOTE (imbalanced-learn)

---

### ✍️ Autor
**Claudio Andrés Díaz Vargas**

🔗 [Ver proyecto completo »](./proyecto1)

---

<hr style="border:1px solid #bbb; margin:40px 0;">

---

## 🏥 Proyecto 2 — Clasificación de Notas Clínicas con Enfoque Ético y Mitigación de Sesgos

> **Evaluación Modular - Módulo 8**  
> **Tema:** Procesamiento de Lenguaje Natural (NLP) aplicado al ámbito clínico.

---

### 🎯 Objetivo
Desarrollar un sistema de **NLP** que clasifique textos médicos según su **gravedad clínica (leve, moderado, severo)**, aplicando buenas prácticas de **preprocesamiento, evaluación y mitigación de sesgos** lingüísticos y sociales.

---

### 🧩 Contexto
Los registros médicos en texto libre contienen información valiosa, pero requieren tiempo y experiencia para analizarse manualmente.  
Este proyecto propone un **sistema automatizado** que asista en la **detección temprana de afecciones**, priorizando pacientes y reduciendo riesgos.

También se analiza la **ética del modelo**, considerando sesgos lingüísticos o sociales, y se incorporan **métodos de interpretabilidad (LIME)** para fortalecer la transparencia del sistema.

---

### 📊 Resumen
Se compararon dos enfoques:

- 📚 **Naive Bayes + TF-IDF**  
- 🤖 **BERT Multilingual / BETO**

Ambos lograron **métricas perfectas en validación**, revelando gran capacidad predictiva pero riesgo de **sobreajuste**.  
Se aplicaron técnicas de **interpretabilidad (LIME)** y una evaluación ética sobre los posibles sesgos en el lenguaje clínico.

---

### 🧰 Tecnologías Utilizadas
- Python  
- Scikit-learn  
- Transformers (Hugging Face)  
- BERT Multilingual / BETO  
- LIME  
- NLTK / spaCy  
- Pandas / NumPy

---

### ✍️ Autor
**Claudio Andrés Díaz Vargas**

🔗 [Ver proyecto completo »](./proyecto2)

---

<hr style="border:1px solid #bbb; margin:40px 0;">

---

## 🔍 Proyecto 3 — Interpretabilidad de Modelos Predictivos usando LIME y SHAP

> **Evaluación Modular - Módulo 9**  
> **Tema:** Explicabilidad y ética en inteligencia artificial.

---

### 🎯 Objetivo
Aplicar herramientas de **explicabilidad de modelos**, como **LIME** y **SHAP**, para analizar y justificar el comportamiento de un modelo de clasificación, destacando la importancia de la transparencia en la inteligencia artificial.

---

### 🧩 Contexto
Formar parte de un equipo que usa **IA en decisiones críticas** implica construir modelos **explicables y éticamente responsables**.  
Este proyecto explora cómo la interpretabilidad permite **auditar la lógica interna** de los modelos y detectar errores o sesgos ocultos.

---

### 📊 Resumen
Se analizó un modelo de **Random Forest** para predecir **enfermedades cardíacas**, utilizando **LIME** y **SHAP**.  
El modelo alcanzó **88.6% de precisión**, pero las explicaciones revelaron **fallas críticas**:

- Uso correcto de variables relevantes (**ECG**, **angina**, **frecuencia cardíaca máxima**).  
- **Manejo incorrecto del colesterol**: interpretó valores bajos como riesgosos y altos como protectores, debido a datos con valores `0.0`.

📉 Sin interpretabilidad, este error habría pasado inadvertido, comprometiendo decisiones clínicas.  
El caso demuestra que **precisión sin transparencia** no es suficiente: la interpretabilidad garantiza **auditoría, confianza y ética** en modelos de IA.

---

### 🧰 Tecnologías Utilizadas
- Python  
- Scikit-learn  
- Random Forest  
- LIME  
- SHAP  
- Matplotlib / Seaborn  
- Pandas / NumPy

---

### ✍️ Autor
**Claudio Andrés Díaz Vargas**

🔗 [Ver proyecto completo »](./proyecto3)

---

<hr style="border:1px solid #bbb; margin:40px 0;">

---

## 🧭 Organización y Buenas Prácticas

Este portafolio está organizado de manera clara y estructurada:

- Navegación sencilla entre secciones  
- Documentación técnica y reflexiva  
- Redacción cuidada, ortografía revisada y estilo profesional  

---
