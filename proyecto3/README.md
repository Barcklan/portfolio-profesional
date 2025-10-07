# 🔍 Proyecto 3 — Interpretabilidad de Modelos Predictivos usando LIME y SHAP

> **Tema:** Explicabilidad y ética en inteligencia artificial

---

## 🎯 Objetivo
Aplicar herramientas de **explicabilidad de modelos**, como **LIME** y **SHAP**, para analizar y justificar el comportamiento de un modelo de clasificación, destacando la importancia de la **transparencia y ética** en IA.

---

## 🧩 Contexto
Formar parte de un equipo que usa **IA en decisiones críticas** implica construir modelos **explicables y auditables**.  
Este proyecto muestra cómo la interpretabilidad permite **detectar errores y sesgos ocultos** y mejorar la confianza en las decisiones automatizadas.

---

## 📊 Resumen del Proyecto
Se analizó un modelo de **Random Forest** para predecir **enfermedades cardíacas**, utilizando **LIME** y **SHAP**:

- Precisión general: **88.6%**  
- Uso correcto de variables relevantes: **ECG, angina, frecuencia cardíaca máxima**  
- Problema detectado: valores de **colesterol mal interpretados** (`0.0`), causando falsos positivos

📈 El proyecto demuestra que **precisión sin interpretabilidad** no es suficiente, y que la transparencia es esencial en IA crítica.

---

## 🧰 Tecnologías Utilizadas
- Python  
- Scikit-learn  
- Random Forest  
- LIME  
- SHAP  
- Matplotlib / Seaborn  
- Pandas / NumPy

---

## 📂 Estructura de Archivos

```bash
 proyecto3/               
   ├─ README.md              ← Carpeta del Proyecto 3
   ├─ data/                 
   ├─ notebooks/             
   ├─ scripts/               
   ├─ reports/              
   └─ requirements.txt       
  ```
