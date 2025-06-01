# topics.py

import io
import re
import boto3
import pandas as pd
import numpy as np

from bertopic import BERTopic

from config import (
    BUCKET,
    TOPICS_PREFIX,    # p.ej. "topicos/playstore"
    SENTIMENT_PREFIX  # p.ej. "sentimientos/playstore"
)

s3 = boto3.client("s3")


# ---------------------------------------------------------
# 1) DICCIONARIO DE CORRECCIONES ORTOGRÁFICAS 
# ---------------------------------------------------------
typo_corrections = {
    "execelente": "excelente",
    "exlecente": "excelente",
    "vien": "bien",
    "trasferencia": "transferencia",
    "tranferencia": "transferencia",
    "ultma": "ultima",
    "ultma_actualizacion": "ultima_actualizacion",
    "abrlr": "abrir",
    "seevicio": "servicio",
    "cervicio": "servicio",
    "bue": "buen",
    "servio": "servicio",
    # …añade aquí otros typos críticos según tu histórico
}
typo_pattern = re.compile(
    r"\b(" + "|".join(map(re.escape, typo_corrections.keys())) + r")\b",
    flags=re.IGNORECASE
)

def correct_typos_once(text: str) -> str:
    def _replacer(match):
        found = match.group(0).lower()
        return typo_corrections.get(found, "")
    return typo_pattern.sub(_replacer, text)


# ---------------------------------------------------------
# 2) NORMALIZACIÓN DE PUNTUACIÓN
# ---------------------------------------------------------
def normalize_punctuation(text: str) -> str:
    # Quita cualquier caracter que no sea letra, número o espacio.
    t = re.sub(r"[^a-z0-9áéíóúñü ]+", " ", text)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------
# 3) STOP-WORDS PARA NEGATIVAS ÚNICAMENTE
# ---------------------------------------------------------
extra_stopwords_neg = {
    "good", "very", "perfect", "super", "thanks", "thank", "like", "cool",
    "awesome", "excellent", "genial", "chido", "chévere", "gracias",
    "nice", "yeah", "great", "you", "that", "doy",
    "fantástico", "fantastica", "fantastico",
    "increíble", "increible", "feliz", "felices",
    "mejor", "recomendable", "recomendada", "recomendado",
    "perfecto", "general"
}
generic_domain_stopwords_neg = {
    "aplicacion", "aplicación", "app", "banco", "bbva", "interfaz",
    "usuario", "usuarios", "login", "sesion", "sesión",
    "transferencias", "pago", "pagos", "funciona", "funcionar", "servicios",
    "bien", "excelente", "bueno", "buena",
    "mal", "mala", "malisimo", "malo",
    "util", "provechoso", "favorable",
    "seguridad", "seguro", "dinero", "movimientos",
    "sirve", "regular", "saca"
}
all_stopwords_neg = extra_stopwords_neg.union(generic_domain_stopwords_neg)
escaped_neg = [re.escape(word) for word in all_stopwords_neg]
stop_pattern_neg = re.compile(r"\b(?:" + "|".join(escaped_neg) + r")\b", flags=re.IGNORECASE)

def remove_stopwords_neg(text: str) -> str:
    cleaned = stop_pattern_neg.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()

# ---------------------------------------------------------
# 4) FUNCIONES AUXILIARES PARA S3 & CARGA
# ---------------------------------------------------------
def get_yyyy_mm_from_s3_prefix() -> str:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=SENTIMENT_PREFIX + "/", Delimiter="/")
    meses = [pref["Prefix"].split("/")[-2] for pref in resp.get("CommonPrefixes", [])]
    if not meses:
        raise RuntimeError(f"No hay carpetas en s3://{BUCKET}/{SENTIMENT_PREFIX}/")
    ultimo_mes = sorted(meses)[-1]
    print(f"🗓️ Último mes detectado en SENTIMENT_PREFIX: {ultimo_mes}")
    return ultimo_mes

def load_sentiment_csv_for_month(month_yyyy_mm: str) -> pd.DataFrame:
    key = f"{SENTIMENT_PREFIX}/{month_yyyy_mm}/reviews_sentiment_{month_yyyy_mm}.csv"
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=["at"])
    print(f"✅ Cargadas {len(df):,} reseñas desde s3://{BUCKET}/{key}")
    return df


# ---------------------------------------------------------
# 5) FUNCIÓN PRINCIPAL: apply_topics()
# ---------------------------------------------------------
def apply_topics():
    # 1) Detectar último mes y cargar CSV de sentimientos
    ultimo_mes = get_yyyy_mm_from_s3_prefix()
    df_sent = load_sentiment_csv_for_month(ultimo_mes)

    # 2) Corregir typos y normalizar puntuación
    df_sent["content_clean"] = (
        df_sent["content_clean"].fillna("")
            .astype(str)
            .str.lower()
            .apply(correct_typos_once)
            .apply(normalize_punctuation)
    )

    # 3) Contar tokens y separar reseñas cortas vs largas
    df_sent["token_count"] = df_sent["content_clean"].str.split().apply(len)
    df_short = df_sent[df_sent["token_count"] < 3].copy()
    df_short["topic_id"] = -1
    df_short["topic_label"] = "Comentario Corto"

    df_long = df_sent[df_sent["token_count"] >= 3].copy()

    # 4) Separar POS y NEG
    df_pos = df_long[df_long["sentiment_pred"] == "pos"].copy()
    df_neg = df_long[df_long["sentiment_pred"] == "neg"].copy()

    print(f"\n→ Del mes {ultimo_mes}: {len(df_sent):,} reseñas totales, "
          f"{len(df_pos):,} POSITIVAS (>=3 palabras), {len(df_neg):,} NEGATIVAS (>=3 palabras), "
          f"{len(df_short):,} CORTAS (<3 palabras).\n")

    # 5) Entrenar BERTopic en POS
    df_pos["topic_id"] = -1
    df_pos["topic_label"] = ""
    df_topics_pos = pd.DataFrame()
    if not df_pos.empty:
        print("=== ENTRENANDO BERTopic sobre POSITIVAS (mes completo) ===")
        modelo_pos = BERTopic(nr_topics=8, calculate_probabilities=True, verbose=False)
        topics_pos, probs_pos = modelo_pos.fit_transform(df_pos["content_clean"].tolist())

        info_pos = modelo_pos.get_topic_info()
        df_topics_pos = pd.DataFrame({
            "topic_id": info_pos["Topic"].astype(int),
            "frequency": info_pos["Count"].astype(int),
            "topic_label": info_pos["Name"].astype(str),
            "score": [
                round(np.array(probs_pos)[np.array(topics_pos) == t, t].mean(), 4)
                if (np.array(topics_pos) == t).sum() > 0 else 0.0
                for t in info_pos["Topic"].astype(int)
            ]
        })
        df_topics_pos.loc[df_topics_pos["topic_id"] == -1, "topic_label"] = "outlier"

        df_pos["topic_id"] = topics_pos
        df_pos = df_pos.merge(
            df_topics_pos[["topic_id", "topic_label"]],
            on="topic_id", how="left"
        )

        print(f"\n--- Resumen Tópicos POS (mes {ultimo_mes}) ---")
        print(df_topics_pos.to_string(index=False))
        print("\n")
    else:
        print("No hay reseñas POSITIVAS (>=3 palabras) para procesar.\n")

    # 6) Entrenar BERTopic en NEG
    df_neg["topic_id"] = -1
    df_neg["topic_label"] = ""
    df_topics_neg = pd.DataFrame()
    if not df_neg.empty:
        print("=== ENTRENANDO BERTopic sobre NEGATIVAS (mes completo) ===")
        # Aplicamos stop-words en un array temporal, sin crear columna
        contenido_neg = df_neg["content_clean"].apply(remove_stopwords_neg).tolist()

        modelo_neg = BERTopic(nr_topics=7, calculate_probabilities=True, verbose=False)
        topics_neg, probs_neg = modelo_neg.fit_transform(contenido_neg)

        info_neg = modelo_neg.get_topic_info()
        df_topics_neg = pd.DataFrame({
            "topic_id": info_neg["Topic"].astype(int),
            "frequency": info_neg["Count"].astype(int),
            "topic_label": info_neg["Name"].astype(str),
            "score": [
                round(np.array(probs_neg)[np.array(topics_neg) == t, t].mean(), 4)
                if (np.array(topics_neg) == t).sum() > 0 else 0.0
                for t in info_neg["Topic"].astype(int)
            ]
        })
        df_topics_neg.loc[df_topics_neg["topic_id"] == -1, "topic_label"] = "outlier"

        df_neg["topic_id"] = topics_neg
        df_neg = df_neg.merge(
            df_topics_neg[["topic_id", "topic_label"]],
            on="topic_id", how="left"
        )

        print(f"\n--- Resumen Tópicos NEG (mes {ultimo_mes}) ---")
        print(df_topics_neg.to_string(index=False))
        print("\n")
    else:
        print("No hay reseñas NEGATIVAS (>=3 palabras) para procesar.\n")

    # 7) Unir df_short + df_pos + df_neg
    df_long_merged = pd.concat([df_pos, df_neg], ignore_index=True)
    df_all = pd.concat([df_short, df_long_merged], ignore_index=True)

    # 8) Guardar CSV fusionado en S3
    out_key = f"{TOPICS_PREFIX}/{ultimo_mes}/topics_{ultimo_mes}.csv"
    buf = io.StringIO()
    df_all.to_csv(buf, index=False, encoding="utf-8")
    s3.put_object(Bucket=BUCKET, Key=out_key, Body=buf.getvalue())
    print(f"✓ CSV de tópicos subido a s3://{BUCKET}/{out_key}")


if __name__ == "__main__":
    apply_topics()