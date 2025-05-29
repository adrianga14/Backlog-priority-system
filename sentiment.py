# sentiment.py

import io
import boto3
import pandas as pd
import joblib

from config import BUCKET, CLEAN_PREFIX, MODEL_KEY_V2, SENTIMENT_PREFIX
# Asegúrate de añadir en config.py:
# SENTIMENT_PREFIX = "sentimientos"

def apply_sentiment():
    """
    1) Detecta el último mes procesado en CLEAN_PREFIX.
    2) Descarga clean_reviews_{ym}.csv desde S3 y lo carga en DataFrame.
    3) Descarga y carga el pipeline balanceado desde S3.
    4) Aplica predict y predict_proba al campo content_clean.
    5) Guarda reviews_sentiment_{ym}.csv en SENTIMENT_PREFIX.
    """
    s3 = boto3.client("s3")

    # 1) Listar carpetas YYYY_MM dentro de CLEAN_PREFIX
    resp  = s3.list_objects_v2(Bucket=BUCKET, Prefix=CLEAN_PREFIX + "/", Delimiter="/")
    meses = [p["Prefix"].split("/")[-2] for p in resp.get("CommonPrefixes", [])]
    if not meses:
        raise RuntimeError(f"No hay carpetas limpias en S3 bajo '{CLEAN_PREFIX}'")

    ultimo_mes = sorted(meses)[-1]
    print(f"🗓️ Último mes CLEAN detectado: {ultimo_mes}")

    # 2) Descargar CSV limpio de ese mes
    clean_key = f"{CLEAN_PREFIX}/{ultimo_mes}/clean_reviews_{ultimo_mes}.csv"
    obj       = s3.get_object(Bucket=BUCKET, Key=clean_key)
    df        = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=["at"])
    print(f"✅ Reseñas limpias cargadas: {len(df):,} filas")

    # 3) Descargar y cargar el modelo balanceado
    tmp_model = "/tmp/model.pkl"
    s3.download_file(BUCKET, MODEL_KEY_V2, tmp_model)
    pipe = joblib.load(tmp_model)
    print("🔍 Modelo balanceado cargado desde S3")

    # 4) Predecir sentimiento
    texts = df["content_clean"].fillna("").astype(str)
    df["sentiment_pred"] = pipe.predict(texts)
    df["prob_pos"]       = pipe.predict_proba(texts)[:, 1]
    print("🔮 Sentimiento aplicado a todas las reseñas")

    # 5) Guardar CSV enriquecido en carpeta SENTIMENT_PREFIX
    out_key = f"{SENTIMENT_PREFIX}/{ultimo_mes}/reviews_sentiment_{ultimo_mes}.csv"
    buf     = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    s3.put_object(Bucket=BUCKET, Key=out_key, Body=buf.getvalue())
    print(f"✓ Predicciones subidas a s3://{BUCKET}/{out_key}")

if __name__ == "__main__":
    apply_sentiment()