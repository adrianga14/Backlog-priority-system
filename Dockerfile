FROM python:3.10-slim

WORKDIR /app
COPY . .

# 1) Dale permisos de lectura y ejecución a todos los .py
RUN find . -type f -name "*.py" -exec chmod 755 {} \;

# 2) Instala deps + RIC
RUN pip install --no-cache-dir \
    boto3 \
    pandas \
    numpy \
    gensim \
    stop-words \
    joblib \
    scikit-learn \
    google_play_scraper \
    awslambdaric

# 3) Runtime de Lambda
ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]