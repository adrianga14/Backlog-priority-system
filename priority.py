# priority.py

import io
import boto3
import pandas as pd

from config import BUCKET, TOPICS_PREFIX, PRIORITY_PREFIX

def compute_priorities():
    s3 = boto3.client("s3")

    # 1) Detectar último mes con topics y descargar CSV
    resp  = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{TOPICS_PREFIX}/", Delimiter="/")
    meses = [p["Prefix"].split("/")[-2] for p in resp.get("CommonPrefixes", [])]
    if not meses:
        raise RuntimeError(f"No hay carpetas bajo {TOPICS_PREFIX}/")

    ultimo_mes = sorted(meses)[-1]
    print(f"🗓️  Último mes TOPICS detectado: {ultimo_mes}")

    topics_key = f"{TOPICS_PREFIX}/{ultimo_mes}/topics_{ultimo_mes}.csv"
    obj        = s3.get_object(Bucket=BUCKET, Key=topics_key)
    df         = pd.read_csv(io.BytesIO(obj["Body"].read()), parse_dates=["at"])
    print(f"✅ Topics cargados: {len(df):,} filas")

    # 2) Agregar nombre legible al topic
    topic_map = {
        0:  "Acceso fallido / login",
        1:  "Transferencias fallidas",
        2:  "App congelada / huella",
        3:  "Pagos / tarjetas",
        4:  "UX dispersa (registro, INE…)",
        5:  "Acceso por sesión bloqueada",
       -1:  "otros"
    }
    df["topic_name"] = df["topic"].map(topic_map).fillna("otros")
    cols = ["topic","topic_name"] + [c for c in df.columns if c not in ("topic","topic_name")]
    df = df[cols]

    # 3) Cálculo de métricas agregadas
    N = 7  # top-N
    agg = (
        df
        .groupby(["topic","topic_name"])
        .agg(
            total_reviews = ("topic", "size"),
            pos_reviews   = ("sentiment_pred", lambda x: (x=="pos").sum()),
            neg_reviews   = ("sentiment_pred", lambda x: (x=="neg").sum()),
            avg_thumbs    = ("thumbsUpCount", "mean")
        )
        .reset_index()
    )
    agg["pos_score"] = agg["pos_reviews"] * agg["avg_thumbs"]
    agg["neg_score"] = agg["neg_reviews"] * agg["avg_thumbs"]

    # 4) Extraer top positivos y negativos
    top_pos = (
        agg
        .sort_values("pos_score", ascending=False)
        .head(N)
        .assign(rank_type="positive")
        .loc[:, ["topic","topic_name","pos_reviews","avg_thumbs","pos_score","rank_type"]]
        .rename(columns={"pos_reviews":"reviews","pos_score":"score"})
    )
    top_neg = (
        agg
        .sort_values("neg_score", ascending=False)
        .head(N)
        .assign(rank_type="negative")
        .loc[:, ["topic","topic_name","neg_reviews","avg_thumbs","neg_score","rank_type"]]
        .rename(columns={"neg_reviews":"reviews","neg_score":"score"})
    )

    # 5) Unir y mostrar
    priority_df = pd.concat([top_pos, top_neg], ignore_index=True)
    print("\n📊 Priority DataFrame:\n")
    print(priority_df.to_string(index=False))

    # 6) Guardar en S3
    out_key = f"{PRIORITY_PREFIX}/{ultimo_mes}/priority_{ultimo_mes}.csv"
    buf     = io.StringIO()
    priority_df.to_csv(buf, index=False, encoding="utf-8")
    s3.put_object(Bucket=BUCKET, Key=out_key, Body=buf.getvalue())
    print(f"\n✓ Priorities guardadas en s3://{BUCKET}/{out_key}")

if __name__ == "__main__":
    compute_priorities()