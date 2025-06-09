# app.py
# ----------------------------------------
# Dashboard de Reseñas + Tópicos con Sentimiento (V3)
# Se añaden:
#   1. Resumen KPI en la parte superior
#   2. Filtrado adicional por calificación o por palabra clave
#   3. Gráfica de distribución de calificaciones
# ----------------------------------------

import streamlit as st
import pandas as pd
import boto3
import io
import altair as alt
from datetime import datetime

from config import BUCKET, TOPICS_PREFIX
# En config.py deben existir:
#    BUCKET = "bbva-playstore-reviews"
#    TOPICS_PREFIX = "topicos/playstore"
aws_access_key_id     = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
aws_region            = st.secrets["aws"]["AWS_DEFAULT_REGION"] 

s3 = boto3.client("s3",
    aws_access_key_id     = aws_access_key_id,
    aws_secret_access_key = aws_secret_access_key,
    region_name           = aws_region
)


# ================================================
# 1) Configuración general de la página
# ================================================
st.set_page_config(
    page_title="Dashboard de Sentimiento y Tópicos (V3)",
    layout="wide",
)

st.title("📊 Dashboard: Sentimiento y Tópicos de Reseñas ")


# ================================================
# 2) Funciones de carga desde S3
# ================================================
@st.cache_data
def list_csv_keys(bucket: str, prefix: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key.lower().endswith(".csv"):
                keys.append(key)
    return keys

@st.cache_data
def load_all_review_topics(bucket: str, prefix: str) -> pd.DataFrame:
    """
    Lee todos los CSV de reseñas + tópicos desde S3 (cada fila es una reseña
    asignada a un tópico; columnas esperadas al menos: review_date, content, score,
    sentiment_pred, topic_id, topic_label, appVersion).
    Devuelve un único DataFrame con todas las reseñas de todos los meses.
    """
 
    keys = list_csv_keys(bucket, prefix)
    dfs = []
    for key in keys:
        content = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        df = pd.read_csv(io.BytesIO(content), on_bad_lines="skip")
        # Validar que existan las columnas mínimas
        required = ["review_date", "content", "score", "sentiment_pred", "topic_id", "topic_label", "appVersion"]
        if not all(col in df.columns for col in required):
            continue
        # Convertir review_date a datetime.date
        df["review_date"] = pd.to_datetime(df["review_date"], format="%Y-%m-%d", errors="coerce").dt.date
        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

# ================================================
# 3) Cargar datos completos de reseñas + tópicos
# ================================================
df_reviews = load_all_review_topics(BUCKET, TOPICS_PREFIX)
if df_reviews.empty:
    st.error("No se encontraron archivos de reseñas+topics en S3 bajo el prefijo indicado.")
    st.stop()

# Bloque 4) Filtro: fecha o una o más versiones
st.subheader("🔍 Filtrar datos")
filter_mode = st.radio(
    "Filtrar por:",
    options=["Rango de fechas", "Versión(es) de la app"],
    index=0
)

# Obtener rango total de fechas y versiones disponibles
min_date = df_reviews["review_date"].min()
max_date = df_reviews["review_date"].max()
versions = df_reviews["appVersion"].dropna().astype(str).unique().tolist()
versions.sort()

if filter_mode == "Rango de fechas":
    start_date, end_date = st.date_input(
        "Selecciona rango de fechas:",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    if start_date > end_date:
        st.error("La fecha inicial no puede ser mayor que la fecha final.")
        st.stop()
    mask = (df_reviews["review_date"] >= start_date) & (df_reviews["review_date"] <= end_date)
    df_range = df_reviews[mask].copy()
    if df_range.empty:
        st.warning("No hay reseñas dentro del rango seleccionado.")
        st.stop()
else:
    selected_versions = st.multiselect(
        "Selecciona una o más versiones de la app:",
        options=versions,
        default=[]
    )
    if not selected_versions:
        st.info("Selecciona al menos una versión para filtrar.")
        st.stop()
    # Filtrar df_reviews por todas las versiones seleccionadas
    df_range = df_reviews[df_reviews["appVersion"].astype(str).isin(selected_versions)].copy()
    if df_range.empty:
        st.warning(f"No se encontraron reseñas para las versiones seleccionadas.")
        st.stop()
    # Mostrar rango de fechas resultante
    start_date = df_range["review_date"].min()
    end_date   = df_range["review_date"].max()
    st.markdown(
        f"**Rango de fechas para versiones seleccionadas:** "
        f"{start_date} – {end_date}"
    )

# ================================================
# 5) Filtros adicionales: selección de calificación mediante estrellas y búsqueda por palabra clave
# ================================================
st.markdown("#### Filtros adicionales")

# 5.1) Select_slider con íconos de estrellas para elegir calificación mínima
#     Opción “Todas” si no se quiere filtrar por calificación
options_stars = [0, 1, 2, 3, 4, 5]
def format_stars(x):
    return "Todas las calificaciones" if x == 0 else "★" * x

min_stars = st.select_slider(
    "Calificación mínima (score):",
    options=options_stars,
    value=0,
    format_func=format_stars
)

if min_stars > 0:
    df_range = df_range[df_range["score"].astype(int) >= min_stars]

# 5.2) Campo de búsqueda por palabra clave en el texto original de la reseña
keyword = st.text_input("🔍 Buscar palabra clave en la reseña:")
if keyword:
    df_range = df_range[df_range["content"].str.contains(keyword, case=False, na=False)]

if df_range.empty:
    st.warning("No hay reseñas que cumplan todos los filtros seleccionados.")
    st.stop()





# ================================================
# 7) Balance de Sentimiento (barra única con porcentajes corregida)
# ================================================
import altair as alt

# Recalcular conteos (usando df_range filtrado)
total_reseñas = df_range.shape[0]
pos_count = df_range[df_range["sentiment_pred"].str.upper() == "POS"].shape[0]
neg_count = df_range[df_range["sentiment_pred"].str.upper() == "NEG"].shape[0]

# Calcular fracción y porcentaje redondeado
pos_frac = pos_count / total_reseñas if total_reseñas else 0
neg_frac = neg_count / total_reseñas if total_reseñas else 0
pos_pct = round(pos_frac * 100, 1)
neg_pct = round(neg_frac * 100, 1)

st.markdown("## 1) Balance de Sentimiento")
st.markdown(
    """
    Muestra cuántas reseñas son positivas y cuántas son negativas,
    junto con sus porcentajes en el conjunto filtrado.
    """
)

# Métricas principales
m1, m2, m3 = st.columns(3)
with m1:
    st.metric(label="Total de reseñas", value=f"{total_reseñas:,}")
with m2:
    st.metric(
        label="Reseñas positivas",
        value=f"{pos_count:,}",
        delta=f"{pos_pct} % del total",
        delta_color="normal",
    )
with m3:
    st.metric(
        label="Reseñas negativas",
        value=f"{neg_count:,}",
        delta=f"{neg_pct} % del total",
        delta_color="inverse",
    )
st.markdown("")
#Graficos de barra
col1, col2 = st.columns([3, 1])

with col1:
    df_bar = pd.DataFrame({
        "Sentimiento": ["Negativo", "Positivo"],
        "Conteo": [neg_count, pos_count]
    })
    bar_chart = (
        alt.Chart(df_bar)
        .mark_bar(size=30)
        .encode(
            y=alt.Y("1:O", axis=None),
            x=alt.X("Conteo:Q", stack="normalize", title="Porcentaje (%)"),
            color=alt.Color("Sentimiento:N", scale=alt.Scale(range=["#e63946", "#2a9d8f"])),
            
        )
        .properties(height=40)
    )
    st.altair_chart(bar_chart, use_container_width=True)
    st.markdown(
        f"<div style='font-size:16px; margin-top:0.5em;'>"
        f"<span style='color:#e63946; font-weight:bold;'>{neg_pct:.1f}% negativo</span>  |  "
        f"<span style='color:#2a9d8f; font-weight:bold;'>{pos_pct:.1f}% positivo</span>"
        f"</div>",
        unsafe_allow_html=True
    )

st.markdown("---")



# ================================================
# 2) Evolución diaria de reseñas POS y NEG + Promedio de calificación
# (sin sección separada de “Calificación General”)
# ================================================
import altair as alt

st.markdown("## 2) Evolución diaria de reseñas")
st.markdown(
    """
   
    """
)

# — Cálculo del promedio general de calificación (sobre todo el rango filtrado)
promedio_general = df_range["score"].mean().round(2)

# — Banner de “Calificación promedio” en amarillo
st.markdown(
    f"""
    <div style="
        background-color: transparent;
        padding: 12px 20px;
        border-radius: 8px;
        margin-bottom: 10px;
        display: inline-block;
    ">
        <span style="font-size:18px; font-weight:500;">
            Calificación promedio:
        </span>
        &nbsp;
        <span style="font-size:32px; color:#FFD700; font-weight:bold;">
            {promedio_general:.2f} ★
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("")  # Espacio antes del gráfico

# — Agrupar por día y sentimiento para conteos POS/NEG
df_daily_sent = (
    df_range
    .groupby(["review_date", "sentiment_pred"])
    .size()
    .reset_index(name="conteo")
)

# — Agrupar por día para promedio de calificación
df_daily_score = (
    df_range
    .groupby("review_date")["score"]
    .mean()
    .reset_index(name="avg_score")
)

# — Crear DataFrame con todas las fechas del rango
all_days = pd.DataFrame({"review_date": pd.date_range(start=start_date, end=end_date)})
all_days["review_date"] = all_days["review_date"].dt.date

# — Construir filas para POS, NEG y promedio
rows = []
for d in all_days["review_date"]:
    # Conteo POS
    pos_row = df_daily_sent[
        (df_daily_sent["review_date"] == d) & 
        (df_daily_sent["sentiment_pred"].str.upper() == "POS")
    ]["conteo"]
    pos_count = int(pos_row.iloc[0]) if not pos_row.empty else 0

    # Conteo NEG
    neg_row = df_daily_sent[
        (df_daily_sent["review_date"] == d) & 
        (df_daily_sent["sentiment_pred"].str.upper() == "NEG")
    ]["conteo"]
    neg_count = int(neg_row.iloc[0]) if not neg_row.empty else 0

    # Promedio de calificación (None si no hay reseñas ese día)
    score_row = df_daily_score[
        df_daily_score["review_date"] == d
    ]["avg_score"]
    avg_score = float(score_row.iloc[0]) if not score_row.empty else None

    # Agregar dos filas: una para POS y otra para NEG, ambas con mismo avg_score
    rows.append({"Fecha": d, "Tipo": "Positivas", "Cantidad": pos_count, "Promedio": avg_score})
    rows.append({"Fecha": d, "Tipo": "Negativas", "Cantidad": neg_count, "Promedio": avg_score})

df_line_all = pd.DataFrame(rows)

# — Gráfico de líneas para POS y NEG
sent_chart = (
    alt.Chart(df_line_all[df_line_all["Tipo"].isin(["Positivas", "Negativas"])])
    .mark_line(point=True, size=2)
    .encode(
        x=alt.X("Fecha:T", title="Fecha"),
        y=alt.Y("Cantidad:Q", title="Cantidad de reseñas"),
        color=alt.Color("Tipo:N", title="Sentimiento",
                        scale=alt.Scale(domain=["Positivas", "Negativas"], 
                                        range=["#2a9d8f", "#e63946"])),
        tooltip=[
            alt.Tooltip("Fecha:T", title="Fecha"),
            alt.Tooltip("Tipo:N", title="Sentimiento"),
            alt.Tooltip("Cantidad:Q", title="Cantidad diaria")
        ]
    )
)

# — Gráfico de línea para promedio de calificación (eje Y secundario)
score_chart = (
    alt.Chart(df_line_all)
    .mark_line(color="#FFD700", strokeDash=[4,2], size=2)
    .encode(
        x=alt.X("Fecha:T", title=""),
        y=alt.Y("Promedio:Q", title="Promedio de calificación", 
                axis=alt.Axis(format=".1f"), scale=alt.Scale(domain=[0, 5])),
        tooltip=[
            alt.Tooltip("Fecha:T", title="Fecha"),
            alt.Tooltip("Promedio:Q", title="Promedio de calif.", format=".2f")
        ]
    )
)

# — Combinar ambos gráficos con escalas Y independientes
combined = (
    alt.layer(sent_chart, score_chart)
       .resolve_scale(y="independent")
       .properties(width="container", height=300)
       .configure_axis(labelFontSize=12, titleFontSize=14)
       .configure_legend(titleFontSize=14, labelFontSize=12)
)

st.altair_chart(combined, use_container_width=True)
st.markdown("---")


# ================================================
#10) Tópicos positivos y negativos (todas, ordenadas por frecuencia)
# ================================================
st.markdown("### 3) Temas mas hablados")

# 1) agregamos conteo por topic y sentimiento
df_topics_agg2 = (
    df_range
    .groupby(["topic_id","topic_label","sentiment_pred"])
    .size()
    .reset_index(name="conteo")
)
df_topics_agg2 = df_topics_agg2[
    ~df_topics_agg2["topic_label"].isin(["outlier","Comentario Corto"])
]

# 2) agregamos la versión más frecuente por tópico/sentimiento
df_version_agg = (
    df_range
    .groupby(["topic_id","sentiment_pred","appVersion"])
    .size()
    .reset_index(name="version_count")
)
df_top_version = (
    df_version_agg
    .sort_values(["topic_id","sentiment_pred","version_count"], ascending=[True,True,False])
    .drop_duplicates(subset=["topic_id","sentiment_pred"])
    .loc[:,["topic_id","sentiment_pred","appVersion"]]
    .rename(columns={"appVersion":"version"})
)

# 3) construir tablas para POS y NEG (sin límite de 5)
df_pos_topics = (
    df_topics_agg2[df_topics_agg2["sentiment_pred"].str.upper()=="POS"]
    .merge(df_top_version[df_top_version["sentiment_pred"].str.upper()=="POS"],
           on=["topic_id","sentiment_pred"], how="left")
    .sort_values("conteo", ascending=False)
    .copy()
)
df_neg_topics = (
    df_topics_agg2[df_topics_agg2["sentiment_pred"].str.upper()=="NEG"]
    .merge(df_top_version[df_top_version["sentiment_pred"].str.upper()=="NEG"],
           on=["topic_id","sentiment_pred"], how="left")
    .sort_values("conteo", ascending=False)
    .copy()
)

# 4) limpiar etiquetas y renombrar columnas de salida
def clean_label(label: str) -> str:
    import re
    text = re.sub(r'^\d+_', '', label)
    return text.replace('_',' ').title()

for df_topics in (df_pos_topics, df_neg_topics):
    df_topics["Tópico"] = df_topics["topic_label"].apply(clean_label)
    df_topics["# de reseñas"] = df_topics["conteo"]
    df_topics["version"] = df_topics["version"].fillna("n/a")

# 5) mostrar en dos columnas
col3, col4 = st.columns(2)
with col3:
    st.markdown("**Temas Positivos**")
    if not df_pos_topics.empty:
        df_display = df_pos_topics[["Tópico","version","# de reseñas"]].reset_index(drop=True)
        styled = df_display.style.set_properties(
            **{"text-align":"left","font-size":"14px","padding":"8px"}
        ).set_table_styles([
            {"selector":"th","props":[("background-color","#333333"),
                                      ("color","white"),("font-size","15px")]}
        ])
        st.dataframe(styled, use_container_width=True, height=300)
    else:
        st.write("No se encontraron reseñas positivas para tópicos en este conjunto.")

with col4:
    st.markdown("**Temas que solucionar**")
    if not df_neg_topics.empty:
        df_display = df_neg_topics[["Tópico","version","# de reseñas"]].reset_index(drop=True)
        styled = df_display.style.set_properties(
            **{"text-align":"left","font-size":"14px","padding":"8px"}
        ).set_table_styles([
            {"selector":"th","props":[("background-color","#333333"),
                                      ("color","white"),("font-size","15px")]}
        ])
        st.dataframe(styled, use_container_width=True, height=300)
    else:
        st.write("No se encontraron reseñas negativas para tópicos en este conjunto.")

st.markdown("---")

# ================================================
# 11) Explorador de reseñas por múltiples tópicos y calificación
# ================================================
st.markdown("### 4) Explora reseñas de uno o más tópicos")

sentiment_choice = st.radio(
    "Mostrar reseñas de:",
    options=["Positivas", "Negativas"],
    index=0
)

mask_sent = df_range["sentiment_pred"].str.upper() == (
    "POS" if sentiment_choice == "Positivas" else "NEG"
)
topics_filtrados = df_range[mask_sent]["topic_label"].unique().tolist()
topics_filtrados = [t for t in topics_filtrados if t not in ["outlier", "Comentario Corto"]]
label_map = {clean_label(t): t for t in topics_filtrados}
lista_limpia = sorted(label_map.keys())

selected_topics_clean = st.multiselect(
    f"Selecciona uno o más tópicos {sentiment_choice.lower()}:",
    options=lista_limpia,
    default=[]
)

if selected_topics_clean:
    selected_topics = [label_map[clean] for clean in selected_topics_clean]
    df_topic_sel = df_range[
        mask_sent & df_range["topic_label"].isin(selected_topics)
    ].copy()
    df_topic_sel["sentimiento_orden"] = df_topic_sel["sentiment_pred"].str.upper().map({"POS": 0, "NEG": 1})
    df_topic_sel = df_topic_sel.sort_values(
        by=["sentimiento_orden", "review_date", "review_time"],
        ascending=[True, False, False]
    ).reset_index(drop=True)
    df_topic_sel.drop(columns="sentimiento_orden", inplace=True)

    if not df_topic_sel.empty:
        st.markdown(
            f"**Reseñas {sentiment_choice.lower()} de tópicos seleccionados "
            f"({', '.join(selected_topics_clean)})**:"
        )
        # añadimos la columna appVersion y la mostramos como "version"
        df_muestra = df_topic_sel[[
            "review_date",
            "review_time",
            "topic_label",
            "content",
            "score",
            "appVersion"
        ]].copy()
        df_muestra.columns = [
            "Fecha",
            "Hora",
            "Tópico",
            "Contenido",
            "Calificación",
            "version"
        ]
        df_muestra["Tópico"] = df_muestra["Tópico"].apply(clean_label)
        st.dataframe(df_muestra, use_container_width=True, height=300)
    else:
        st.write("No hay reseñas para esos tópicos y calificación en el conjunto filtrado.")
else:
    st.info("Selecciona uno o más tópicos disponibles según el sentimiento elegido.")

# FIN DE app.py
