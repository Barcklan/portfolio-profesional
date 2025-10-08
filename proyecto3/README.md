# 🔍 Proyecto 3 — Interpretabilidad de Modelos Predictivos usando LIME y SHAP

> ### Explicabilidad y ética en inteligencia artificial

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
   ├─ 📜 README.md              ← Carpeta del Proyecto 3
   ├─ 📂 data/                 
   ├─ 📔 notebooks/             
   ├─ 📂 scripts/               
   ├─ 📂 reports/              
   └─ 📜 requirements.txt       
  ```
--- 

## 📈 Resultados Principales/Hallazgos

El análisis de un modelo de Random Forest para predecir enfermedades cardíacas utilizando herramientas de interpretabilidad como SHAP y LIME ha revelado que, aunque el modelo tiene una alta precisión general (88.6%), su lógica interna es defectuosa y potencialmente peligrosa. La interpretabilidad demostró que el modelo se apoya en variables clínicamente válidas y de peso, como los patrones de ECG durante el ejercicio, la angina y la frecuencia cardíaca máxima. Sin embargo, también se descubrió una falla crítica: el modelo interpreta de manera inconsistente y anómala la variable Colesterol, tratando los valores bajos como un factor de riesgo significativo y los altos como protectores.

Esta inconsistencia, probablemente causada por errores de entrada de datos (valores de 0.0), llevó a predicciones incorrectas en casos específicos, generando falsos positivos a pesar de que el paciente no presentaba una enfermedad cardíaca. Este proyecto subraya que la precisión no es suficiente en áreas críticas como la salud. Sin interpretabilidad, no se podría auditar la lógica del modelo, identificar sus sesgos y vulnerabilidades, o comprender por qué falla en casos particulares, lo que demuestra que la transparencia y la responsabilidad son esenciales para la implementación segura y ética de la inteligencia artificial.

A futuro, se planeará una mejora del modelo de Random Forest, enfocada en una depuración y validación exhaustiva de los datos de entrada, así como en la optimización de los hiperparámetros y la evaluación comparativa con otros algoritmos como XGBoost o modelos basados en redes neuronales. Además, se buscará incorporar un proceso continuo de monitorización del rendimiento y reentrenamiento con nuevos datos clínicos, garantizando así un modelo más robusto, confiable y alineado con los principios de la medicina basada en evidencia.

<p align="center">
  <img src="img/Metricas.png" width="45.7%" />
  <img src="img/F1-Score.png" width="45%" />
</p>

<div align="center">

 <H3> Modelo Random Forest entrenado </H3>
  <H4> Métricas en prueba: <b>accuracy: 0.89</b> </H4>
|Estado Cardíaco Paciente| `precision` | `recall` | `F1-Score` |
|-----------|-----------|-----------|-----------|
|Sano (=0)| 0.89 | 0.85 | 0.87 |
|Enfermo(=1)| 0.89 | 0.91 | 0.90 |


</div>



Tanto Naive Bayes como BERT alcanzaron un rendimiento perfecto (100% en precisión, recall, F1 y accuracy), lo que sugiere que el dataset es pequeño y fácilmente separable, con posible sobreajuste. No se observa ventaja entre ambos modelos: Naive Bayes es más rápido y eficiente para tareas simples, mientras que BERT ofrece mayor robustez para escenarios más complejos o con mayor volumen de datos.

### Explicabilidad con LIME:

Dado que ambos modelos —Naive Bayes y BERT— alcanzaron un rendimiento perfecto (1.00 en accuracy, precision, recall y F1-score), resulta fundamental analizar cómo y por qué llegan a sus predicciones. La interpretabilidad mediante LIME (Local Interpretable Model-agnostic Explanations) permite comprender qué palabras o patrones lingüísticos influyen más en la clasificación de la gravedad clínica del paciente (leve, moderado o severo).

A través de LIME, se busca verificar si las decisiones de los modelos son coherentes con el contexto médico, identificar posibles errores de interpretación semántica y garantizar que la alta precisión observada no oculte sesgos o sobreajuste hacia ciertas clases o términos clínicos.

<div align="center">
  <H3> Naive Bayes </H3>
</div>
<p align="center">
  <img src="img/LIME_BN.png" width="60%" />
</p>

El modelo Naive Bayes clasificó el texto como “leve” con un 98% de probabilidad. LIME evidenció que palabras como complicaciones o severas, aunque sugieren mayor gravedad, fueron interpretadas por el modelo como asociadas a casos leves, descartando así las clases “moderado” y “severo”.

<div align="center">
  <H3>BERT</H3> 
</div>
<p align="center">
  <img src="img/LIME_BERT.png" width="80%" />
</p>

El modelo BERT clasificó el texto como “severo” (52%) al identificar términos clave como dificultad respiratoria, hospitalización y requiere inmediata, asociados a alta gravedad. Aunque las clases “leve” y “moderado” tuvieron cierta probabilidad, las palabras clínicas reforzaron la decisión hacia “severo”.


## 📄 Conclusiones

El proyecto demostró que las técnicas de NLP pueden clasificar eficazmente notas clínicas según la gravedad del paciente. Naive Bayes ofreció una línea base interpretable, mientras que BERT logró mayor comprensión semántica y precisión. El uso de LIME aportó transparencia al mostrar las palabras clave que influyen en las predicciones. Además, se destacó la necesidad de abordar sesgos, privacidad y supervisión médica, promoviendo un uso ético y responsable de la IA como herramienta de apoyo en la detección y priorización clínica.


#### 🔗 [Ver análisis completo en el Notebook (.ipynb) »](./notebooks/CNCEE_NLP_clean.ipynb)
