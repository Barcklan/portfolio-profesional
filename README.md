# 💼 Portafolio Profesional

## 👋 Sobre mí

<img src="https://github.com/barcklan.png" width="120" align="left" style="border-radius:50%; margin-right:20px;">

Soy un profesional apasionado por el análisis de datos, la modelación estadística y el desarrollo de soluciones basadas en *Machine Learning* que contribuyan a la toma de decisiones informadas.


**Nombre:** Claudio Díaz Vargas

**Especialización:** Ingeniero en Estadística, especializado en Machine Learning y Ciencia de Datos.

**Correo:** cdiazv.ies@gmail.com

**GitHub:** [https://github.com/barcklan](https://github.com/barcklan)

Soy un profesional apasionado por el análisis de datos, la modelación estadística y el desarrollo de soluciones basadas en Machine Learning que contribuyan a la toma de decisiones informadas.

Tengo experiencia en análisis exploratorio de datos, modelado estadístico, técnicas de Machine Learning supervisado y no supervisado, optimización de modelos, visualización de datos y automatización de procesos analíticos con Python.

Mi objetivo es aplicar mis conocimientos en estadística y Machine Learning para diseñar soluciones basadas en datos que apoyen la toma de decisiones y generen impacto positivo.

---

## 🚀 Proyectos Destacados

A continuación, se presentan tres de mis proyectos más relevantes, donde aplico mis conocimientos en análisis, programación y documentación técnica.

---

### **PROYECTO 1 - SISTEMA INTELIGENTE DE SCORING CREDITICIO CON REDES NEURONALES PROFUNDAS (DNN)**

---

### 🎯 Objetivo

Diseñar, entrenar y evaluar un modelo de red neuronal profunda para predecir la probabilidad de impago de clientes bancarios, utilizando un conjunto de datos realista.  
El modelo debe ser explicable, eficiente y presentar resultados interpretables para su uso en contextos financieros.

### 🧠 Contexto

Las entidades financieras deben decidir si otorgan o no un crédito a un cliente.  
Esta decisión, si se basa en modelos poco explicables, puede generar sesgos, exclusiones injustas o pérdidas económicas.  

Se busca construir un modelo moderno, basado en **redes neuronales profundas**, que sea a la vez **preciso y explicable**, permitiendo a las instituciones mejorar la calidad de sus decisiones crediticias.

### 📊 Resumen

Se desarrolló un sistema de **scoring crediticio** basado en **redes neuronales profundas (DNN)** para predecir la probabilidad de impago de clientes bancarios.

El proyecto comenzó con un **análisis exploratorio** que identificó diferencias claras entre buenos y malos pagadores, especialmente en **monto y duración del crédito**.  
Tras un proceso de **preprocesamiento de datos** y la aplicación de **SMOTE** para balancear las clases, se entrenaron dos modelos principales:

- **DNN simple:** desempeño superior, con *accuracy* de **70,5%** y **AUC = 0,78**, mostrando buena generalización y equilibrio entre clientes *good* y *bad*.  
- **ResNet tabular:** obtuvo *accuracy* de **64%** y **AUC = 0,64**, con mejor detección de clientes *bad* pero mayor riesgo de falsos positivos.

📈 En conclusión, la **DNN simple** se posiciona como la opción más confiable y efectiva para decisiones de crédito, aunque aún puede mejorarse la predicción de clientes solventes y la interpretabilidad del modelo.

### 🧰 Tecnologías Utilizadas

- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **SMOTE (imbalanced-learn)**

### ✍️ Autor

**Claudio Andrés Díaz Vargas**

🔗 [Ver proyecto completo](./proyecto1)

---

### **PROYECTO 2 - CLASIFICACIÓN DE NOTAS CLÍNICAS PARA DETECCIÓN TEMPRANA DE AFECCIONES**  
### **CON ENFOQUE ÉTICO Y MITIGACIÓN DE SESGOS**

---

### 🎯 Objetivo

Desarrollar un sistema de **procesamiento de lenguaje natural (NLP)** capaz de clasificar textos médicos —como notas clínicas, síntomas o diagnósticos— según su **gravedad clínica (leve, moderado, severo)**.  

El modelo debe integrar **buenas prácticas de preprocesamiento, evaluación y mitigación de sesgos lingüísticos y sociales**, garantizando transparencia y responsabilidad ética en su uso.

### 🧠 Contexto

Los registros médicos en texto libre contienen información valiosa para detectar la gravedad de una afección de forma temprana. Sin embargo, su interpretación manual requiere tiempo, conocimiento médico y puede verse afectada por **sesgos humanos**.

En este proyecto se desarrolla un sistema **automatizado de NLP** que analiza y clasifica notas clínicas según su **nivel de gravedad clínica**. Este sistema puede asistir a profesionales de la salud en la **priorización de pacientes**, mejorando la eficiencia del sistema sanitario y reduciendo riesgos.

Además, se realiza un análisis crítico de los **posibles sesgos lingüísticos o sociales** que puedan influir en el modelo, junto con el uso de **técnicas de interpretabilidad** para asegurar la confianza en su aplicación clínica.

### 📊 Resumen

Se implementaron y compararon dos enfoques principales:

- **Naive Bayes con TF-IDF:** enfoque clásico basado en estadísticas de frecuencia de palabras.  
- **BERT en español:** modelo de lenguaje contextualizado de última generación.

Ambos fueron entrenados sobre un **dataset de notas clínicas** y lograron **métricas perfectas en validación**, evidenciando tanto la alta capacidad de separación de los datos como un posible **riesgo de sobreajuste**.  

Para asegurar la **transparencia y explicabilidad**, se aplicaron métodos como **LIME** y se realizó una evaluación ética de **riesgos y sesgos** potenciales en las predicciones.

En conjunto, el proyecto demuestra cómo los sistemas de NLP pueden aplicarse en contextos clínicos de manera **efectiva y responsable**, siempre considerando los aspectos éticos y sociales de la inteligencia artificial.

### 🧰 Tecnologías Utilizadas

- **Python**
- **Scikit-learn**
- **Transformers (Hugging Face)**
- **BERT Multilingual / BETO**
- **LIME**
- **NLTK / spaCy**
- **Pandas / NumPy**

## ✍️ Autor

**Claudio Andrés Díaz Vargas**

🔗 [Ver proyecto completo](./proyecto2)

---

### **PROYECTO 3 - INTERPRETABILIDAD DE MODELOS PREDICTIVOS USANDO LIME Y SHAP**

---

### 🎯 Objetivo

Aplicar herramientas de **explicabilidad de modelos**, específicamente **LIME** y **SHAP**, para analizar y justificar el comportamiento de un modelo de clasificación, destacando la importancia de la transparencia en modelos de inteligencia artificial.

### 🧠 Contexto

Imagina formar parte de un equipo de ciencia de datos en una organización que utiliza **inteligencia artificial para apoyar decisiones críticas**.  
Estas decisiones deben ser **comprensibles para personas no técnicas** —clientes, médicos, auditores o usuarios finales—.  

Por ello, el desafío no consiste únicamente en construir un modelo preciso, sino en garantizar que sea **explicable, auditable y éticamente responsable**.

### 📊 Resumen

Se analizó un modelo de **Random Forest** para predecir **enfermedades cardíacas**, utilizando herramientas de interpretabilidad como **SHAP** y **LIME**.

El modelo mostró una **alta precisión general (88.6%)**, pero la interpretabilidad reveló fallos importantes:

- El modelo se apoyaba correctamente en variables **clínicamente válidas**, como:
  - Patrones de **ECG durante el ejercicio**
  - **Angina inducida por esfuerzo**
  - **Frecuencia cardíaca máxima alcanzada**

- Sin embargo, se detectó una **inconsistencia crítica**:  
  el modelo interpretaba los **valores bajos de colesterol** como **mayor riesgo**,  
  y los **valores altos como protectores**, lo cual es clínicamente incorrecto.

Este comportamiento fue atribuido a **errores en los datos de entrada**, donde algunos valores de colesterol aparecían como `0.0`, distorsionando el aprendizaje del modelo y provocando **falsos positivos** (predicciones de enfermedad inexistente).

📈 Este caso demuestra que la **precisión por sí sola no es suficiente** en ámbitos sensibles como la salud.  
La **interpretabilidad** permite auditar la lógica interna del modelo, identificar sesgos y vulnerabilidades, y comprender sus fallos, garantizando una **IA segura, ética y transparente**.

### 🧰 Tecnologías Utilizadas

- **Python**
- **Scikit-learn**
- **Random Forest**
- **LIME**
- **SHAP**
- **Matplotlib / Seaborn**
- **Pandas / NumPy**

### ✍️ Autor

**Claudio Andrés Díaz Vargas**

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
🔗 [https://github.com/tuusuario/portfolio-profesional](https://github.com/barcklan/portfolio-profesional)
