# topics.py

import io
import os
import boto3
import pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary

from config import (
    BUCKET,
    CLEAN_PREFIX,     # p.ej. "clean/playstore"
    TOPIC_MODEL_KEY,  # p.ej. "models/lda.model"
    DICT_KEY,         # p.ej. "models/lda.dict"
    TOPICS_PREFIX,    # p.ej. "topicos/playstore"
    SENTIMENT_PREFIX, # p.ej. "sentimientos/playstore"
)

def apply_topics():
    s3 = boto3.client("s3")

    # 1) Descargar y cargar LDA + diccionario en /tmp/lda_model
    tmp = "/tmp/lda_model"
    os.makedirs(tmp, exist_ok=True)
    artifacts = {
        TOPIC_MODEL_KEY:                       f"{tmp}/lda.model",
        TOPIC_MODEL_KEY + ".state":           f"{tmp}/lda.model.state",
        TOPIC_MODEL_KEY + ".expElogbeta.npy": f"{tmp}/lda.model.expElogbeta.npy",
        TOPIC_MODEL_KEY + ".id2word":         f"{tmp}/lda.model.id2word",
        DICT_KEY:                              f"{tmp}/lda.dict",
    }
    for s3_key, local in artifacts.items():
        print(f"↓ Descargando s3://{BUCKET}/{s3_key} → {local}")
        s3.download_file(BUCKET, s3_key, local)

    lda        = LdaModel.load(f"{tmp}/lda.model")
    dictionary = Dictionary.load(f"{tmp}/lda.dict")
    print(f"🔍 LDA cargado: num_topics={lda.num_topics}, vocab_size={len(dictionary)}")

    # 2) Detectar último mes en SENTIMENT_PREFIX
    resp   = s3.list_objects_v2(Bucket=BUCKET, Prefix=SENTIMENT_PREFIX + "/", Delimiter="/")
    meses  = [p["Prefix"].split("/")[-2] for p in resp.get("CommonPrefixes", [])]
    if not meses:
        raise RuntimeError(f"No hay carpetas en s3://{BUCKET}/{SENTIMENT_PREFIX}/")
    ultimo_mes = sorted(meses)[-1]
    print(f"🗓️ Último mes SENTIMENT detectado: {ultimo_mes}")

    # 3) Descargar CSV de sentimientos de ese mes
    sent_key = f"{SENTIMENT_PREFIX}/{ultimo_mes}/reviews_sentiment_{ultimo_mes}.csv"
    obj      = s3.get_object(Bucket=BUCKET, Key=sent_key)
    df       = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=["at"])
    print(f"✅ Cargadas {len(df):,} reseñas desde s3://{BUCKET}/{sent_key}")

    # 4) Asignar topic dominante
    def dominant_topic(text: str) -> int:
        bow = dictionary.doc2bow(str(text).split())
        if not bow:
            return -1
        topic_id, _ = max(lda.get_document_topics(bow), key=lambda x: x[1])
        return topic_id

    df["topic"] = df["content_clean"].fillna("").astype(str).apply(dominant_topic)
    print("🏷️ Tópicos asignados")

    # 5) Guardar CSV con tópicos en S3
    out_key = f"{TOPICS_PREFIX}/{ultimo_mes}/topics_{ultimo_mes}.csv"
    buf     = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    s3.put_object(Bucket=BUCKET, Key=out_key, Body=buf.getvalue())
    print(f"✓ CSV de tópicos subido a s3://{BUCKET}/{out_key}")

if __name__ == "__main__":
    apply_topics()