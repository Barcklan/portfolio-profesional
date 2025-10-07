# 🧠 Proyecto 1 — Sistema Inteligente de Scoring Crediticio (DNN)

> **Evaluación Modular - Módulo 7**  
> **Tema:** Redes Neuronales Profundas aplicadas a riesgo crediticio

---

## 🎯 Objetivo
Diseñar, entrenar y evaluar un modelo de red neuronal profunda para predecir la probabilidad de impago de clientes bancarios, utilizando un conjunto de datos realista.  
El modelo debe ser explicable, eficiente y presentar resultados interpretables para su uso en contextos financieros.

---

## 🧩 Contexto
Las entidades financieras deben decidir si otorgan o no un crédito a un cliente.  
Si la decisión se basa en modelos poco explicables, puede generar sesgos, exclusiones injustas o pérdidas económicas.  

Este proyecto busca construir un modelo **moderno, preciso y explicable**, basado en **redes neuronales profundas (DNN)**, que permita mejorar la calidad de las decisiones crediticias.

---

## 📊 Resumen del Proyecto
- Se realizó un **análisis exploratorio de datos (EDA)** para identificar patrones de buen/mal pagador.  
- Se aplicó **preprocesamiento** y balanceo de clases con **SMOTE**.  
- Se entrenaron dos modelos principales:
  - 🧠 **DNN simple** → Accuracy: 70.5%, AUC: 0.78  
  - ⚙️ **ResNet tabular** → Accuracy: 64%, AUC: 0.64  

📈 La **DNN simple** mostró mejor generalización y equilibrio, siendo la opción más confiable.

---

## 🧰 Tecnologías Utilizadas
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas / NumPy  
- Matplotlib / Seaborn  
- SMOTE (imbalanced-learn)

---

## 📂 Estructura de Archivos

```bash
 proyecto1/                 ← Carpeta del Proyecto 1
   ├─ README.md              ← Documentación detallada del proyecto 1
   ├─ data/                  ← (Opcional) datasets utilizados
   ├─ notebooks/             ← Notebooks de análisis y entrenamiento
   ├─ scripts/               ← Scripts Python (.py) de entrenamiento, preprocesamiento, etc.
   ├─ reports/               ← Gráficos, visualizaciones, resultados
   └─ requirements.txt       ← Dependencias del proyecto
```
