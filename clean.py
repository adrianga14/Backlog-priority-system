# clean.py

import io
import re
import boto3
import pandas as pd
from stop_words import get_stop_words
from unicodedata import normalize
from config import BUCKET, RAW_PREFIX, CLEAN_PREFIX

# Cliente S3
s3 = boto3.client("s3")


def clean_new_reviews(ym: str):
    """
    1) Descarga raw/playstore/{ym}/reviews_{ym}.csv
    2) Elimina columnas userName, userImage, reviewCreatedVersion, replyContent, repliedAt
    2.5) Procesar fecha: convertir 'at' a datetime y separar fecha y hora
    3) Normaliza texto (quita acentos y stop-words)
    4) Guarda clean/{ym}/clean_reviews_{ym}.csv
    """
    # 1) cargar CSV raw
    raw_key = f"{RAW_PREFIX}/{ym}/reviews_{ym}.csv"
    obj     = s3.get_object(Bucket=BUCKET, Key=raw_key)
    df      = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=["at"])

    # 2.5) Procesar fecha: convertir 'at' a datetime y separar fecha y hora
    df['at'] = pd.to_datetime(df['at'])
    df['review_date'] = df['at'].dt.date
    df['review_time'] = df['at'].dt.time

    # 2) elimina columnas que NO queremos
    df = df.drop(columns=["userName", "userImage", "reviewCreatedVersion", "replyContent", "repliedAt"], errors="ignore")

    # 3) función de limpieza
    sw = set(get_stop_words("spanish"))
    def _clean(txt):
        txt = str(txt).lower()
        # descompone acentos: e.g. á → a +  ́
        txt = normalize("NFKD", txt)
        # quita marcas de acento, conserva la letra
        txt = "".join(ch for ch in txt if not re.match(r'[\u0300-\u036f]', ch))
        # elimina puntuación/símbolos
        txt = re.sub(r"[^\w\s]", " ", txt)
        # tokeniza y filtra stop-words y tokens muy cortos
        toks = [w for w in txt.split() if w not in sw and len(w) > 2]
        return " ".join(toks)

    df["content_clean"] = df["content"].apply(_clean)

    # 4) guardar limpio en S3
    out_key = f"{CLEAN_PREFIX}/{ym}/clean_reviews_{ym}.csv"
    buf     = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    s3.put_object(Bucket=BUCKET, Key=out_key, Body=buf.getvalue())
    print(f"✓ Datos limpios guardados en s3://{BUCKET}/{out_key}  ({len(df):,} filas)")


def main():
    # detecta último mes en raw/playstore/
    resp   = s3.list_objects_v2(Bucket=BUCKET, Prefix=RAW_PREFIX + "/", Delimiter="/")
    meses  = [p["Prefix"].split("/")[-2] for p in resp.get("CommonPrefixes", [])]
    if not meses:
        raise RuntimeError("No hay carpetas en raw/playstore/")

    ultimo_mes = sorted(meses)[-1]
    print(f"🗓️  Último mes RAW detectado: {ultimo_mes}")
    clean_new_reviews(ultimo_mes)


if __name__ == "__main__":
    main()
