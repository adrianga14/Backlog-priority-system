
 # 🧠 Backlog Priority System
 
 Este proyecto automatiza la recolección, limpieza, análisis de sentimiento y priorización de reseñas de apps móviles desde Play Store, con el objetivo de generar insumos accionables para el backlog de producto.
 
 ---
 
 ## 🚀 Objetivo
 
 Convertir las reseñas crudas en datos estructurados y limpios que permitan:
 - Detectar sentimientos positivos/negativos/neutros
 - Agrupar temáticas comunes
 - Priorizar problemas críticos con base en frecuencia y severidad
 
 ---
 
 
 ## 📋 Estructura del proyecto

Backlog-priority-system

	•	clean.py — Limpieza de texto, fechas y columnas irrelevantes
	•	extract.py — Extracción de datos crudos desde S3
	•	sentiment.py — Clasificación de sentimientos (modelo simple)
	•	topics.py — Modelado de temas con LDA/BERT
	•	priority.py — Cálculo de prioridad por frecuencia x sentimiento
	•	priority.py (obsoleto) — Calculaba prioridad solo por frecuencia
	•	orchestrator.py — Orquestador que ejecuta el pipeline completo
	•	app.py — Dashboard interactivo de sentimiento y tópicos
	•	config.py — Rutas S3 y configuración central
	•	requirements.txt — Dependencias necesarias
 ---
 
 ## 📦 Dependencias
 
 - Python 3.9+
 - Pandas
 - Boto3
 - Stop-words
 - Scikit-learn
 - BERTopic (para modelado de temas)
 - Streamlit (si se usa para visualización)
 
 Instalación:
 ```bash
 pip install -r requirements.txt
 ```
 
 ## 🧪 Ejecución local
 ```bash
 python clean.py       # Limpia y guarda las reseñas procesadas en S3
 python sentiment.py   # Clasifica sentimientos
 python topics.py      # Genera tópicos desde el texto limpio
-python priority.py    # Calcula prioridades
+python orchestrator.py  # Ejecuta el pipeline completo
 ```
 
 ## ☁️ Integración con AWS
 
 El sistema está diseñado para ejecutarse automáticamente en la nube usando:
 - AWS Lambda para cada etapa del pipeline
 - Amazon S3 como almacenamiento fuente y destino
 - AWS EventBridge para agendar la ejecución cada lunes a las 6:00 AM (hora CDMX)
 
+## 🎛️ Dashboard
+
+El archivo `app.py` genera un dashboard en Streamlit disponible en [https://backlogprioritysystem.streamlit.app/](https://backlogprioritysystem.streamlit.app/).
+Ofrece:
+- Filtro por rango de fechas o versión de la app
+- Búsqueda por palabra clave y calificación mínima
+- Métricas de sentimiento y evolución diaria
+- Top tópicos positivos y negativos
+- Explorador detallado de reseñas por tópico
+
 ## 🔖 Versiones
 
-La versión actual es v1.0, que incluye el sistema base de limpieza y procesamiento automático de reseñas.
+La versión actual es **v2.0**. Esta versión introduce el orquestador del pipeline, publica el dashboard interactivo y elimina el paso de `priority.py` porque calculaba prioridad solo por frecuencia.
+
+El número de versión también se guarda en el archivo `VERSION` y el repositorio cuenta con la etiqueta git `v2.0`.
 
 Consulta el CHANGELOG para más detalles.
 
 ## 👨‍💻 Autor
 
 Adrián Galicia  · 2025
 
 ## 📄 Licencia
 
 Este repositorio es privado y propiedad de Adrián Galicia. Todos los derechos reservados.
