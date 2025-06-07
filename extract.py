# extract.py

import io
import pandas as pd
import boto3
import time
from datetime import datetime, timedelta, timezone
from google_play_scraper import reviews, Sort

from config import APP_ID, BUCKET, RAW_PREFIX, WINDOW_DAYS

# Zona horaria CDMX
TZ_MX = timezone(timedelta(hours=-6))

# Parámetros de paginación
LOTE       = 1000    # reseñas por llamada
PAUSA_S    = 0.2     # segundos entre llamadas
MAX_VACIOS = 3       # para cortar si no vienen más filas

def extract_reviews():
    """
    Descarga reseñas de los últimos WINDOW_DAYS días y las sube a S3
    en raw/playstore/YYYY_MM/reviews_YYYY_MM.csv según su mes de publicación.
    """
    # 1) Ventana de fechas
    end_dt   = datetime.now(TZ_MX)
    start_dt = end_dt - timedelta(days=WINDOW_DAYS)
    print(f"🔍 Extrayendo reseñas entre {start_dt.date()} y {end_dt.date()}...")

    all_rows = []
    token    = None
    vacios   = 0
    seen     = set()

    # 2) Paginación y filtrado manual
    while True:
        filas, token_next = reviews(
            APP_ID,
            lang="es",
            country="mx",
            sort=Sort.NEWEST,
            count=LOTE,
            continuation_token=token
        )

        stop_early = False
        for f in filas:
            fecha = f["at"].astimezone(TZ_MX)
            if fecha < start_dt:
                stop_early = True
                break
            if fecha <= end_dt:
                all_rows.append(f)

        if stop_early:
            break

        if not filas:
            vacios += 1
            if vacios >= MAX_VACIOS:
                print("   ⚠️  lotes vacíos consecutivos, paro.")
                break
        else:
            vacios = 0

        if not token_next or token_next in seen:
            break

        seen.add(token_next)
        token = token_next
        time.sleep(PAUSA_S)

    # 3) DataFrame y eliminación de columnas no deseadas
    df = pd.DataFrame(all_rows)
    if df.empty:
        print("⚠️  No se encontraron reseñas en este rango.")
        return

    df = df.drop(columns=["userName", "userImage", "reviewCreatedVersion", "replyContent", "repliedAt"], errors="ignore")

    # 4) Agrupar por mes y subir CSVs
    df["mes"] = pd.to_datetime(df["at"]).dt.strftime("%Y_%m")
    s3 = boto3.client("s3")

    for ym, grupo in df.groupby("mes"):
        key = f"{RAW_PREFIX}/{ym}/reviews_{ym}.csv"
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=key)
            prev = pd.read_csv(io.BytesIO(obj["Body"].read()))
            print(f"   • Archivo existente encontrado para {ym}, fusionando...")
        except s3.exceptions.NoSuchKey:
            prev = pd.DataFrame()
        except Exception as e:
            print(f"   ⚠️  Error al leer archivo existente: {e}")
            prev = pd.DataFrame()

        # Fusionar sin duplicados por reviewId
        if not prev.empty:
            merged = pd.concat([prev, grupo.drop(columns=["mes"])], ignore_index=True)
            merged = merged.drop_duplicates(subset=["reviewId"])
        else:
            merged = grupo.drop(columns=["mes"])

        buf = io.StringIO()
        merged.to_csv(buf, index=False, encoding="utf-8")
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
        print(f"✓ {len(merged):,} reseñas subidas → s3://{BUCKET}/{key}")

if __name__ == "__main__":
    extract_reviews()

