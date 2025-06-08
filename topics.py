# topics.py
import os
import io
import re
import boto3
import pandas as pd
import numpy as np

from bertopic import BERTopic

from config import BUCKET, TOPICS_PREFIX, SENTIMENT_PREFIX

s3 = boto3.client("s3")


# ---------------------------------------------------------
# 1) LISTAR MESES DISPONIBLES EN S3
# ---------------------------------------------------------
def list_available_months() -> list[str]:
    resp = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=SENTIMENT_PREFIX + "/",
        Delimiter="/"
    )
    meses = [p["Prefix"].split("/")[-2] for p in resp.get("CommonPrefixes", [])]
    if not meses:
        raise RuntimeError(f"No hay carpetas en s3://{BUCKET}/{SENTIMENT_PREFIX}/")
    return sorted(meses)


# ---------------------------------------------------------
# 2) CARGAR CSV DE SENTIMIENTO PARA UN MES
# ---------------------------------------------------------
def load_sentiment_csv_for_month(yyyy_mm: str) -> pd.DataFrame:
    key = f"{SENTIMENT_PREFIX}/{yyyy_mm}/reviews_sentiment_{yyyy_mm}.csv"
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=["at"])
    print(f"✅ Cargadas {len(df):,} reseñas desde s3://{BUCKET}/{key}")
    return df


# ---------------------------------------------------------
# 3) SELECCIONAR ÚLTIMO MES CON ≥300 RESEÑAS (o retroceder)
# ---------------------------------------------------------
def select_month_with_min_reviews(min_reviews: int = 300) -> tuple[str, pd.DataFrame]:
    meses = list_available_months()  # e.g. ["2025_03","2025_04","2025_05"]
    for mes in reversed(meses):
        df = load_sentiment_csv_for_month(mes)
        if len(df) >= min_reviews:
            print(f"→ Seleccionado {mes}: {len(df)} reseñas (>= {min_reviews})")
            return mes, df
        print(f"⚠️ {mes} tiene solo {len(df)} reseñas (< {min_reviews}), pruebo mes anterior...")
    # si ninguno cumple, devolver el más antiguo
    mes_ant = meses[0]
    df_ant = load_sentiment_csv_for_month(mes_ant)
    print(f"⚠️ Ningún mes con ≥{min_reviews}, usando más antiguo {mes_ant} ({len(df_ant)} reseñas)")
    return mes_ant, df_ant


# ---------------------------------------------------------
# 4) TYPOS Y NORMALIZACIÓN
# ---------------------------------------------------------
typo_corrections = {
    "execelente": "excelente", "exlecente": "excelente",
    "vien": "bien", "trasferencia": "transferencia",
    "tranferencia": "transferencia", "ultma": "ultima",
    "ultma_actualizacion": "ultima_actualizacion", "abrlr": "abrir",
    "seevicio": "servicio", "cervicio": "servicio",
    "bue": "buen", "servio": "servicio"
}
typo_pattern = re.compile(
    r"\b(" + "|".join(map(re.escape, typo_corrections.keys())) + r")\b",
    flags=re.IGNORECASE
)
def correct_typos_once(text: str) -> str:
    return typo_pattern.sub(lambda m: typo_corrections[m.group(0).lower()], text)

def normalize_punctuation(text: str) -> str:
    t = re.sub(r"[^a-z0-9áéíóúñü ]+", " ", text)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------
# 5) STOP-WORDS PARA POS & NEG
# ---------------------------------------------------------
extra_stopwords_neg = {
    "good","very","perfect","super","thanks","thank","like","cool",
    "awesome","excellent","genial","chido","chévere","gracias",
    "nice","yeah","great","you","that","doy","fantástico","fantastica",
    "fantastico","increíble","increible","feliz","felices","mejor",
    "recomendable","recomendada","recomendado","perfecto","general",
    "facil","usar","apps","eee","love","banca","bancaria","ohh"
}
generic_domain_stopwords_neg = {
    "aplicacion","aplicación","app","banco","bbva","interfaz",
    "usuario","usuarios","login","sesion","sesión","transferencias",
    "pago","pagos","funciona","funcionar","servicios","bien",
    "excelente","bueno","buena","mal","mala","malisimo","malo",
    "util","provechoso","favorable","seguridad","seguro","dinero",
    "movimientos","sirve","regular","saca"
}
all_stopwords_neg = extra_stopwords_neg.union(generic_domain_stopwords_neg)
stop_pattern_neg = re.compile(
    r"\b(?:" + "|".join(map(re.escape, all_stopwords_neg)) + r")\b",
    flags=re.IGNORECASE
)
def remove_stopwords_neg(text: str) -> str:
    cleaned = stop_pattern_neg.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


# ---------------------------------------------------------
# 6) PUNTO CENTRAL: apply_topics()
# ---------------------------------------------------------
def apply_topics():
    # 6.a) Elegir mes y cargar datos
    mes, df = select_month_with_min_reviews(min_reviews=300)

    # 6.b) Limpieza de texto
    df["content_clean"] = (
        df["content_clean"].fillna("")
          .astype(str)
          .str.lower()
          .apply(correct_typos_once)
          .apply(normalize_punctuation)
    )
    df["token_count"] = df["content_clean"].str.split().apply(len)

    # 6.c) Separar cortas vs largas
    df_short = df[df["token_count"] < 3].copy()
    df_short["topic_id"] = -1
    df_short["topic_label"] = "Comentario Corto"
    df_long = df[df["token_count"] >= 3].copy()

    # 6.d) POS vs NEG
    df_pos = df_long[df_long["sentiment_pred"] == "pos"].copy()
    df_neg = df_long[df_long["sentiment_pred"] == "neg"].copy()

    print(
        f"\n→ Mes {mes}: {len(df):,} totales, "
        f"{len(df_pos):,} POS, {len(df_neg):,} NEG, "
        f"{len(df_short):,} CORTAS.\n"
    )

    # 6.e) BERTopic en POS (6 tópicos)
    if not df_pos.empty:
        print("=== ENTRENANDO BERTopic sobre POSITIVAS ===")
        docs_pos = df_pos["content_clean"].apply(remove_stopwords_neg).tolist()
        model_pos = BERTopic(nr_topics=20, calculate_probabilities=True, verbose=False)
        topics_pos, probs_pos = model_pos.fit_transform(docs_pos)

        info_pos = model_pos.get_topic_info()
        df_topics_pos = pd.DataFrame({
            "topic_id":    info_pos["Topic"].astype(int),
            "frequency":   info_pos["Count"].astype(int),
            "topic_label": info_pos["Name"].astype(str),
            "score": [
                round(np.array(probs_pos)[np.array(topics_pos)==t, t].mean(), 4)
                if (np.array(topics_pos)==t).sum()>0 else 0.0
                for t in info_pos["Topic"].astype(int)
            ]
        })
        df_topics_pos.loc[df_topics_pos["topic_id"]==-1, "topic_label"] = "outlier"
        print(df_topics_pos.to_string(index=False), "\n")

        df_pos["topic_id"] = topics_pos
        df_pos = df_pos.merge(
            df_topics_pos[["topic_id","topic_label"]],
            on="topic_id", how="left"
        )
    else:
        print("No hay reseñas POSITIVAS (>=3 palabras)\n")

    # 6.f) BERTopic en NEG
    if not df_neg.empty:
        print("=== ENTRENANDO BERTopic sobre NEGATIVAS ===")
        docs_neg = df_neg["content_clean"].apply(remove_stopwords_neg).tolist()
        model_neg = BERTopic(nr_topics=30, calculate_probabilities=True, verbose=False)
        topics_neg, probs_neg = model_neg.fit_transform(docs_neg)

        info_neg = model_neg.get_topic_info()
        df_topics_neg = pd.DataFrame({
            "topic_id":    info_neg["Topic"].astype(int),
            "frequency":   info_neg["Count"].astype(int),
            "topic_label": info_neg["Name"].astype(str),
            "score": [
                round(np.array(probs_neg)[np.array(topics_neg)==t, t].mean(), 4)
                if (np.array(topics_neg)==t).sum()>0 else 0.0
                for t in info_neg["Topic"].astype(int)
            ]
        })
        df_topics_neg.loc[df_topics_neg["topic_id"]==-1, "topic_label"] = "outlier"
        print(df_topics_neg.to_string(index=False), "\n")

        df_neg["topic_id"] = topics_neg
        df_neg = df_neg.merge(
            df_topics_neg[["topic_id","topic_label"]],
            on="topic_id", how="left"
        )
    else:
        print("No hay reseñas NEGATIVAS (>=3 palabras)\n")

    # 6.g) Unir y subir
    df_all = pd.concat([df_short, df_pos, df_neg], ignore_index=True)
    out_key = f"{TOPICS_PREFIX}/{mes}/topics_{mes}.csv"
    buf = io.StringIO()
    df_all.to_csv(buf, index=False, encoding="utf-8")
    s3.put_object(Bucket=BUCKET, Key=out_key, Body=buf.getvalue())
    print(f"✓ CSV de tópicos subido a s3://{BUCKET}/{out_key}")


if __name__ == "__main__":
    apply_topics()
