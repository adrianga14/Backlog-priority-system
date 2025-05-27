# 🧠 Backlog Priority System

Este proyecto automatiza la recolección, limpieza, análisis de sentimiento y priorización de reseñas de apps móviles desde Play Store, con el objetivo de generar insumos accionables para el backlog de producto.

---

## 🚀 Objetivo

Convertir las reseñas crudas en datos estructurados y limpios que permitan:
- Detectar sentimientos positivos/negativos/neutros
- Agrupar temáticas comunes
- Priorizar problemas críticos con base en frecuencia y severidad

---

## 🛠️ Estructura del Proyecto
Backlog-priority-system/
│
├── clean.py           # Limpieza de texto, fechas y columnas irrelevantes
├── extract.py         # Extracción de datos crudos desde S3
├── sentiment.py       # Clasificación de sentimientos (modelo simple)
├── topics.py          # Modelado de temas con LDA/BERT
├── priority.py        # Cálculo de prioridad por frecuencia x sentimiento
├── config.py          # Rutas S3 y configuración central
└── requirements.txt   # Dependencias necesarias
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
python priority.py    # Calcula prioridades
```

## ☁️ Integración con AWS

El sistema está diseñado para ejecutarse automáticamente en la nube usando:
- AWS Lambda para cada etapa del pipeline
- Amazon S3 como almacenamiento fuente y destino
- AWS EventBridge para agendar la ejecución cada lunes a las 6:00 AM (hora CDMX)

## 🔖 Versiones

La versión actual es v1.0, que incluye el sistema base de limpieza y procesamiento automático de reseñas.

Consulta el CHANGELOG para más detalles.

## 👨‍💻 Autor

Adrián Galicia  · 2025

## 📄 Licencia

Este repositorio es privado y propiedad de Adrián Galicia. Todos los derechos reservados.