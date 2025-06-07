# orchestrator.py

from extract import extract_reviews
from clean import main as clean_main
from sentiment import apply_sentiment
from topics import apply_topics
from config import PIPELINE_VERSION

# Nota: se eliminó el uso de `priority.py` ya que la prioridad se calculaba
# únicamente por frecuencia. El análisis ahora se realiza en el dashboard.

def run_pipeline():
    """
    Función central que ejecuta todo el flujo del pipeline:
    1) Extrae reseñas
    2) Limpia texto
    3) Aplica análisis de sentimientos
    4) Detecta tópicos
    """
    try:
        print(f"🟡 Iniciando pipeline v{PIPELINE_VERSION}...")

        print("➡️ Extrayendo reseñas...")
        extract_reviews()

        print("➡️ Limpiando texto...")
        clean_main()

        print("➡️ Aplicando sentimiento...")
        apply_sentiment()

        print("➡️ Detectando tópicos...")
        apply_topics()

        print("✅ Pipeline ejecutado correctamente.")
        return {
            "statusCode": 200,
            "body": "Pipeline ejecutado correctamente"
        }

    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        return {
            "statusCode": 500,
            "body": str(e)
        }

def lambda_handler(event=None, context=None):
    """
    Handler oficial para AWS Lambda.
    """
    return run_pipeline()

# 🔁 Permite ejecutar el pipeline directamente si se corre localmente
if __name__ == "__main__":
    run_pipeline()