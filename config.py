# config.py

# Versión actual del pipeline (sincronizada con el archivo VERSION)
PIPELINE_VERSION = "2.0"

# — Play Store app ID —
APP_ID     = "com.bancomer.mbanking"

# — S3 bucket y prefijos —
BUCKET = "bbva-playstore-reviews"
RAW_PREFIX     = "raw/playstore"     # lugar donde se suben los CSV mensuales originales
CLEAN_PREFIX   = "clean/playstore"   # lugar donde se suben los CSV limpios
SENTIMENT_PREFIX = "sentimientos"    # lugar donde se suben los CSV con sentimiento

# — Fechas dinámicas —
START_DATE   = None  # ya no se usan: extracción por ventana
END_DATE     = None

# — Fase 1: extracción rolling window —
WINDOW_DAYS  = 7     # cuantos días atrás extraer

# — Fase 2: limpieza —
# (en el .py o notebook, esto se recalcula con cada mes detectado)
RAW_CLEAN_KEY = f"{CLEAN_PREFIX}/reviews_new.csv"

# — Fase 3: sentimiento —
MODEL_KEY     = "models/model_logreg_bal.pkl"  # pipeline balanceado en S3
MODEL_KEY_V2 = "models/model_logreg_bal_v2.pkl"  # modelo binario (pos/neg)


# — Fase 3 output —
ENRICHED_KEY  = f"{CLEAN_PREFIX}/reviews_with_sentiment.csv"


SENTIMENT_PREFIX = "sentimientos"
TOPICS_PREFIX    = "topicos"
TOPIC_MODEL_KEY = "models/lda.model"       # metadatos del modelo
DICT_KEY        = "models/lda.dict"        # diccionario gensim

PRIORITY_PREFIX = "prioridad/playstore"

TOPICS_PREFIX   = "topicos/playstore"
PRIORITY_PREFIX = "prioridad/playstore"