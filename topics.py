# topics.py
import os
import io
import re
import boto3
import pandas as pd
import numpy as np

from datetime import datetime
from dateutil.relativedelta import relativedelta

from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
from hdbscan import HDBSCAN

from config import (
    BUCKET,
    TOPICS_PREFIX,    # ejemplo: "topicos/playstore"
    SENTIMENT_PREFIX  # ejemplo: "sentimientos/playstore"
)

s3 = boto3.client("s3")


# ---------------------------------------------------------
# 1) OBTENER TODOS LOS MESES DISPONIBLES EN S3 BAJO SENTIMENT_PREFIX
# ---------------------------------------------------------
def list_available_months() -> list[str]:
    """
    Lista todos los subprefijos (por ejemplo, "2025_05", "2025_06", etc.)
    bajo SENTIMENT_PREFIX/ en S3 y los devuelve ordenados cronológicamente.
    """
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=SENTIMENT_PREFIX + "/", Delimiter="/")
    meses = [pref["Prefix"].split("/")[-2] for pref in resp.get("CommonPrefixes", [])]
    if not meses:
        raise RuntimeError(f"No hay carpetas en s3://{BUCKET}/{SENTIMENT_PREFIX}/")
    # Ordenamos como strings "YYYY_MM" produce orden cronológico natural
    meses_ordenados = sorted(meses)
    return meses_ordenados


# ---------------------------------------------------------
# 2) CARGAR EL CSV DE SENTIMIENTOS DE S3 PARA UN MES DADO
# ---------------------------------------------------------
def load_sentiment_csv_for_month(month_yyyy_mm: str) -> pd.DataFrame:
    """
    Descarga y retorna el DataFrame para reviews_sentiment_{month_yyyy_mm}.csv.
    """
    key = f"{SENTIMENT_PREFIX}/{month_yyyy_mm}/reviews_sentiment_{month_yyyy_mm}.csv"
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=["at"])
    print(f"✅ Cargadas {len(df):,} reseñas desde s3://{BUCKET}/{key}")
    return df


# ---------------------------------------------------------
# 3) SELECCIONAR EL MES MÁS RECIENTE CON AL MENOS 300 RESEÑAS
# ---------------------------------------------------------
def select_month_with_min_reviews(min_reviews: int = 300) -> tuple[str, pd.DataFrame]:
    """
    Recorre los meses disponibles (de más reciente a más antiguo).
    - Si encuentra un mes con >= min_reviews, lo retorna junto con su DataFrame.
    - Si ninguno cumple, retorna el mes más antiguo disponible (aunque tenga < min_reviews).
    """
    meses = list_available_months()           # e.g. ["2025_03", "2025_04", "2025_05"]
    # Recorremos de más reciente a más antiguo:
    for mes in reversed(meses):
        df = load_sentiment_csv_for_month(mes)
        if len(df) >= min_reviews:
            print(f"→ Elegido mes {mes} porque tiene {len(df)} reseñas (>= {min_reviews}).")
            return mes, df

        print(f"⚠️ El mes {mes} solo tiene {len(df)} reseñas (< {min_reviews}). Bajamos al anterior...")
    # Si ninguno cumple, tomamos el mes más antiguo disponible:
    mes_antiguo = meses[0]
    df_antiguo = load_sentiment_csv_for_month(mes_antiguo)
    print(
        f"⚠️ Ningún mes tiene ≥ {min_reviews} reseñas. "
        f"Tomando el mes más antiguo: {mes_antiguo} con {len(df_antiguo)} reseñas."
    )
    return mes_antiguo, df_antiguo


# ---------------------------------------------------------
# 4) DICCIONARIO DE CORRECCIONES ORTOGRÁFICAS / TYPOS
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
# 5) NORMALIZACIÓN DE PUNTUACIÓN
# ---------------------------------------------------------
def normalize_punctuation(text: str) -> str:
    t = re.sub(r"[^a-z0-9áéíóúñü ]+", " ", text)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------
# 6) STOP-WORDS PARA NEGATIVAS ÚNICAMENTE
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
# 7) DESCARGA RECURSIVA DEL EMBEDDING MODEL LOCAL DESDE S3
# ---------------------------------------------------------
def download_s3_prefix(bucket: str, prefix: str, local_dir: str):
    """
    Descarga todos los objetos bajo 'prefix' en el bucket S3 hacia 'local_dir',
    respetando la estructura de carpetas.
    """
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(prefix):].lstrip("/")
            if rel_path == "":
                continue
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)


# ---------------------------------------------------------
# 8) FUNCIÓN PRINCIPAL: apply_topics()
# ---------------------------------------------------------
def apply_topics():
    # ——————————————————————————————
    # 8.a) DESCARGAR EL EMBEDDING MODEL DESDE S3
    # ——————————————————————————————
    local_model_dir = "/tmp/all-MiniLM-L6-v2"
    s3_model_prefix = (
        "models/embeddings/all-MiniLM-L6-v2/"
        "models--sentence-transformers--all-MiniLM-L6-v2/"
        "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/"
    )
    os.makedirs(local_model_dir, exist_ok=True)
    download_s3_prefix(BUCKET, s3_model_prefix, local_model_dir)

    # Verifico que exista config.json en /tmp/all-MiniLM-L6-v2
    if not os.path.exists(os.path.join(local_model_dir, "config.json")):
        raise RuntimeError("No encontré config.json en /tmp/all-MiniLM-L6-v2 tras descargar.")

    # ——————————————————————————————
    # 8.b) SELECCIONAR EL MES ADECUADO CON AL MENOS 300 RESEÑAS
    # ——————————————————————————————
    mes, df_sent = select_month_with_min_reviews(min_reviews=300)

    # ——————————————————————————————
    # 8.c) LIMPIEZA DE TEXTO: corrección de typos + normalización
    # ——————————————————————————————
    df_sent["content_clean"] = (
        df_sent["content_clean"].fillna("")
            .astype(str)
            .str.lower()
            .apply(correct_typos_once)
            .apply(normalize_punctuation)
    )

    # ——————————————————————————————
    # 8.d) CONTAR TOKENS Y SEPARAR RESEÑAS CORTAS (<3 palabras) VS LARGAS
    # ——————————————————————————————
    df_sent["token_count"] = df_sent["content_clean"].str.split().apply(len)
    df_short = df_sent[df_sent["token_count"] < 3].copy()
    df_short["topic_id"] = -1
    df_short["topic_label"] = "Comentario Corto"

    df_long = df_sent[df_sent["token_count"] >= 3].copy()

    # ——————————————————————————————
    # 8.e) SEPARAR POSITIVAS vs NEGATIVAS
    # ——————————————————————————————
    df_pos = df_long[df_long["sentiment_pred"] == "pos"].copy()
    df_neg = df_long[df_long["sentiment_pred"] == "neg"].copy()

    print(
        f"\n→ Del mes {mes}: {len(df_sent):,} reseñas totales, "
        f"{len(df_pos):,} POSITIVAS (>=3 palabras), "
        f"{len(df_neg):,} NEGATIVAS (>=3 palabras), "
        f"{len(df_short):,} CORTAS (<3 palabras).\n"
    )

       # ——————————————————————————————
    # 8.f) ENTRENAR BERTopic SOBRE POSITIVAS
    # ——————————————————————————————
    df_topics_pos = pd.DataFrame()
    if not df_pos.empty:
        print("=== ENTRENANDO BERTopic sobre POSITIVAS (mes completo) ===")

        # 1) TF-IDF para fallback/híbrido (opcional)
        vectorizer_pos = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        docs_pos = df_pos["content_clean"].tolist()
        X_pos_tfidf = vectorizer_pos.fit_transform(docs_pos)
        emb_pos_tfidf = X_pos_tfidf.toarray()

        # —————— A partir de aquí ya no hay parámetros dinámicos ——————

        # 2) UMAP con valores fijos
        umap_model_pos = umap.UMAP(
            n_neighbors=15,        # valor fijo
            n_components=5,
            metric="cosine",
            low_memory=True,
            random_state=42
        )

        # 3) HDBSCAN con valores fijos
        hdbscan_model_pos = HDBSCAN(
            min_cluster_size=15,   # valor fijo
            min_samples=15,        # valor fijo
            prediction_data=True
        )

        # 4) Instanciar BERTopic
        modelo_pos = BERTopic(
            embedding_model=local_model_dir,
            vectorizer_model=None,    # desactivo TF-IDF interno
            umap_model=umap_model_pos,
            hdbscan_model=hdbscan_model_pos,
            nr_topics=8,
            calculate_probabilities=True,
            verbose=False
        )

        # 5) Entrenamiento
        topics_pos, probs_pos = modelo_pos.fit_transform(docs_pos)

        # 6) Construir DataFrame de resumen para POS
        info_pos = modelo_pos.get_topic_info()
        probs_array_pos = np.array(probs_pos)
        topics_array_pos = np.array(topics_pos)

        scores_pos = []
        for t in info_pos["Topic"].astype(int):
            if probs_array_pos.ndim == 1:
                # Si viene 1D (muy pocas muestras), asignamos 0.0
                scores_pos.append(0.0)
            else:
                mask = (topics_array_pos == t)
                if mask.sum() > 0:
                    mean_val = probs_array_pos[mask, t].mean()
                    scores_pos.append(round(mean_val, 4))
                else:
                    scores_pos.append(0.0)

        df_topics_pos = pd.DataFrame({
            "topic_id": info_pos["Topic"].astype(int),
            "frequency": info_pos["Count"].astype(int),
            "topic_label": info_pos["Name"].astype(str),
            "score": scores_pos
        })
        df_topics_pos.loc[df_topics_pos["topic_id"] == -1, "topic_label"] = "outlier"

        df_pos["topic_id"] = topics_pos
        df_pos = df_pos.merge(
            df_topics_pos[["topic_id", "topic_label"]],
            on="topic_id", how="left"
        )

        print(f"\n--- Resumen Tópicos POS (mes {mes}) ---")
        print(df_topics_pos.to_string(index=False))
        print("\n")
    else:
        print("No hay reseñas POSITIVAS (>=3 palabras) para procesar.\n")


    # ——————————————————————————————
    # 8.g) ENTRENAR BERTopic SOBRE NEGATIVAS
    # ——————————————————————————————
    df_topics_neg = pd.DataFrame()
    if not df_neg.empty:
        print("=== ENTRENANDO BERTopic sobre NEGATIVAS (mes completo) ===")

        contenido_neg = df_neg["content_clean"].apply(remove_stopwords_neg).tolist()
        vectorizer_neg = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        X_neg_tfidf = vectorizer_neg.fit_transform(contenido_neg)
        emb_neg_tfidf = X_neg_tfidf.toarray()

        # —————— UMAP con valores fijos ——————
        umap_model_neg = umap.UMAP(
            n_neighbors=15,        # valor fijo
            n_components=5,
            metric="cosine",
            low_memory=True,
            random_state=42
        )

        # —————— HDBSCAN con valores fijos ——————
        hdbscan_model_neg = HDBSCAN(
            min_cluster_size=15,   # valor fijo
            min_samples=15,        # valor fijo
            prediction_data=True
        )

        modelo_neg = BERTopic(
            embedding_model=local_model_dir,
            vectorizer_model=None,
            umap_model=umap_model_neg,
            hdbscan_model=hdbscan_model_neg,
            nr_topics=7,
            calculate_probabilities=True,
            verbose=False
        )

        topics_neg, probs_neg = modelo_neg.fit_transform(contenido_neg)

        info_neg = modelo_neg.get_topic_info()
        probs_array_neg = np.array(probs_neg)
        topics_array_neg = np.array(topics_neg)

        scores_neg = []
        for t in info_neg["Topic"].astype(int):
            if probs_array_neg.ndim == 1:
                scores_neg.append(0.0)
            else:
                mask = (topics_array_neg == t)
                if mask.sum() > 0:
                    mean_val = probs_array_neg[mask, t].mean()
                    scores_neg.append(round(mean_val, 4))
                else:
                    scores_neg.append(0.0)

        df_topics_neg = pd.DataFrame({
            "topic_id": info_neg["Topic"].astype(int),
            "frequency": info_neg["Count"].astype(int),
            "topic_label": info_neg["Name"].astype(str),
            "score": scores_neg
        })
        df_topics_neg.loc[df_topics_neg["topic_id"] == -1, "topic_label"] = "outlier"

        df_neg["topic_id"] = topics_neg
        df_neg = df_neg.merge(
            df_topics_neg[["topic_id", "topic_label"]],
            on="topic_id", how="left"
        )

        print(f"\n--- Resumen Tópicos NEG (mes {mes}) ---")
        print(df_topics_neg.to_string(index=False))
        print("\n")
    else:
        print("No hay reseñas NEGATIVAS (>=3 palabras) para procesar.\n")


    # ——————————————————————————————
    # 8.h) UNIR df_short + df_pos + df_neg Y GUARDAR EN S3
    # ——————————————————————————————
    df_short["topic_id"] = -1
    df_short["topic_label"] = "Comentario Corto"
    df_long_merged = pd.concat([df_pos, df_neg], ignore_index=True)
    df_all = pd.concat([df_short, df_long_merged], ignore_index=True)

    out_key = f"{TOPICS_PREFIX}/{mes}/topics_{mes}.csv"
    buf = io.StringIO()
    df_all.to_csv(buf, index=False, encoding="utf-8")
    s3.put_object(Bucket=BUCKET, Key=out_key, Body=buf.getvalue())
    print(f"✓ CSV de tópicos subido a s3://{BUCKET}/{out_key}")


if __name__ == "__main__":
    apply_topics()