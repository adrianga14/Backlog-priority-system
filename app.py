# app.py
# Dashboard de Prioridad de Reseñas - Streamlit

import streamlit as st
import pandas as pd
import boto3
import io
import altair as alt
from config import PRIORITY_PREFIX, TOPICS_PREFIX

# ================================================
# 1) Configuración de la página
# ================================================
st.set_page_config(
    page_title="Priority System Dashboard",
    layout="wide",
)

# ================================================
# 2) Funciones de carga de datos
# ================================================
@st.cache_data
def list_csv_keys(bucket: str, prefix: str) -> list[str]:
    """
    Obtiene la lista de objetos .csv en un bucket S3 bajo un prefijo.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key.lower().endswith(".csv"):
                keys.append(key)
    return keys

@st.cache_data
def load_all_csvs(bucket: str, prefix: str) -> pd.DataFrame:
    """
    Descarga y concatena todos los CSVs encontrados en S3.
    Añade columna 'month' a partir del nombre de archivo (formato YYYY_MM).
    """
    keys = list_csv_keys(bucket, prefix)
    dfs = []
    s3 = boto3.client("s3")
    for key in keys:
        content = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        df = pd.read_csv(io.BytesIO(content), on_bad_lines="skip")
        parts = key.split("/")[-1].replace(".csv", "").split("_")
        df["month"] = f"{parts[-2]}_{parts[-1]}"
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ================================================
# 3) Carga de datos desde S3
# ================================================
BUCKET = "bbva-playstore-reviews"
df_prio_all = load_all_csvs(BUCKET, PRIORITY_PREFIX)
df_topics_all = load_all_csvs(BUCKET, TOPICS_PREFIX)

# ================================================
# 4) Sidebar: selección de filtros
# ================================================
months = sorted(df_prio_all.get("month", pd.Series(dtype=str)).unique(), reverse=True)
selected_month = st.sidebar.selectbox("Mes actual", months)
comparison_month = st.sidebar.selectbox(
    "Mes comparación", months, index=1 if len(months) > 1 else 0
)
keyword = st.sidebar.text_input("Buscar palabra clave", "")
if st.sidebar.button("Limpiar filtros"):
    st.experimental_rerun()

# Filtrar DataFrames
# Crear DataFrames solo para el mes seleccionado
df_month_prio = df_prio_all[df_prio_all["month"] == selected_month]
df_month_topics = df_topics_all[df_topics_all["month"] == selected_month]

# ================================================
# 5) Cálculo de métricas
# ================================================
# 5.1 Prioridad global
total_all = len(df_prio_all)
neg_all = df_prio_all[df_prio_all.get("rank_type") == "negative"].shape[0]
pos_all = df_prio_all[df_prio_all.get("rank_type") == "positive"].shape[0]

# 5.2 Topics global
total_topics = df_topics_all["content"].count()
neg_topics = df_topics_all[df_topics_all["sentiment_pred"] == "neg"].shape[0]
pos_topics = df_topics_all[df_topics_all["sentiment_pred"] == "pos"].shape[0]
neu_topics = df_topics_all[df_topics_all["sentiment_pred"] == "neu"].shape[0]

# 5.3 Mes actual
total_month = len(df_month_prio)
neg_month = df_month_prio[df_month_prio.get("rank_type") == "negative"].shape[0]
pos_month = df_month_prio[df_month_prio.get("rank_type") == "positive"].shape[0]

# ================================================
# 6) Header principal
# ================================================
st.title("🚀 Priority System Dashboard")
st.write("Análisis automatizado de reseñas y priorización")

# ================================================
# 7) Bloque I: Sentimientos globales (topics)
# ================================================
col1, col2 = st.columns([3, 2])
with col1:
    neg_pct = neg_topics / total_topics if total_topics else 0
    pos_pct = pos_topics / total_topics if total_topics else 0
    neu_pct = neu_topics / total_topics if total_topics else 0

    st.subheader("Sentimientos globales de Topics")
    st.write(
        f"**{neg_pct*100:.0f}% Negativo** | "
        f"**{pos_pct*100:.0f}% Positivo** | "
        f"**{neu_pct*100:.0f}% Neutro**"
    )
    df_bar = pd.DataFrame({
        "Sentimiento": ["Negativo", "Positivo", "Neutro"],
        "Porcentaje": [neg_pct, pos_pct, neu_pct]
    })
    bar_chart = (
        alt.Chart(df_bar)
        .mark_bar()
        .encode(
            x=alt.X("Porcentaje:Q", stack="normalize", axis=None),
            color=alt.Color("Sentimiento:N", scale=alt.Scale(range=["#e63946", "#2a9d8f", "#aaaaaa"]))
        )
        .properties(height=30)
    )
    st.altair_chart(bar_chart, use_container_width=True)
    st.caption("Distribución de sentimiento en todos los topics")
with col2:
    st.subheader("Filtros")
    st.write(f"Mes actual: {selected_month}")
    st.write(f"Mes comparación: {comparison_month}")
    st.write(f"Keyword: {keyword}")

# ================================================
# 8) Bloque II: Resumen global de Topics
# ================================================
st.subheader("Resumen de Topics (Global)")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Total topics", f"{total_topics:,}")
t2.metric("Negativas", f"{neg_topics:,}")
t3.metric("Positivas", f"{pos_topics:,}")
t4.metric("Neutrales", f"{neu_topics:,}")

# ================================================
# 9) Bloque III: Evolución de Sentimiento
# ================================================
st.subheader("Sentiment Over Time")
if not df_prio_all.empty:
    time_df = df_prio_all[df_prio_all["month"].isin([selected_month, comparison_month])]
    agg = (
        time_df.groupby(["month", "rank_type"]).size()
        .reset_index(name="count")
        .pivot(index="month", columns="rank_type", values="count").reset_index()
    ).fillna(0)
    agg["total"] = agg["negative"] + agg["positive"]
    for col in ["negative", "positive"]:
        agg[col] = agg[col] / agg["total"]
    melt_df = agg.melt(
        id_vars=["month"], value_vars=["negative", "positive"], var_name="Sentiment", value_name="Value"
    )
    line_chart = (
        alt.Chart(melt_df)
        .mark_line(point=True)
        .encode(x="month:N", y=alt.Y("Value:Q", axis=alt.Axis(format="%")), color="Sentiment:N")
        .properties(height=300)
    )
    st.altair_chart(line_chart, use_container_width=True)
else:
    st.info("No hay datos para mostrar la evolución del sentimiento.")

# ================================================
# 10) Bloque IV: Cosas más/menos gustadas (mes actual)
# ================================================
st.subheader("Cosas más/menos gustadas (Mes actual)")
if not df_month_prio.empty:
    df_m = df_month_prio.copy()
    if keyword:
        df_m = df_m[df_m["topic_name"].str.contains(keyword, case=False, na=False)]
    top_pos = df_m[df_m["rank_type"] == "positive"].nlargest(3, "score")["topic_name"].tolist()
    top_neg = df_m[df_m["rank_type"] == "negative"].nlargest(3, "score")["topic_name"].tolist()
else:
    top_pos, top_neg = [], []
colp, cold = st.columns(2)
with colp:
    st.write("**Top Positivos**")
    for topic in top_pos:
        st.write(f"- {topic}")
with cold:
    st.write("**Top Negativos**")
    for topic in top_neg:
        st.write(f"- {topic}")

# ================================================
# 11) Bloque V: Explorador de reseñas detalladas
# ================================================
st.subheader("Reviews Explorer")
if (
    not df_month_topics.empty
    and set(["content", "score", "appVersion", "sentiment_pred"]).issubset(df_month_topics.columns)
):
    st.dataframe(
        df_month_topics[["content", "score", "appVersion", "sentiment_pred"]],
        use_container_width=True
    )
else:
    st.info("No hay datos de topics para el mes seleccionado o faltan columnas esperadas.")


