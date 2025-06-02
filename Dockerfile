FROM public.ecr.aws/lambda/python:3.10

WORKDIR ${LAMBDA_TASK_ROOT}

RUN yum update -y \
 && yum install -y \
      gcc gcc-c++ \
      python3-devel \
      cmake \
      make \
      libjpeg-turbo-devel \
      zlib-devel \
      freetype-devel \
      pkgconfig \
      openblas-devel \
      lapack-devel \
      boost-devel \
 && yum clean all

# Instalamos todas las librerías con versiones exactas
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir \
       torch==2.1.0\
       torchvision==0.16.0\
       torchaudio==2.1.0\
       --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir \
       transformers==4.52.4 \
       bertopic==0.17.0 \
       umap-learn==0.5.2 \
       hdbscan==0.8.32 \
       sentence-transformers==4.1.0 \
       scikit-learn==1.2.2 \
       joblib==1.5.0 \
       numpy==1.25.2 \
       pandas==2.1.4 \
       boto3==1.35.99 \
       altair==5.0.1 \
       plotly==6.0.1 \

       wordcloud==1.9.2 \
       stop-words==2018.7.23 \
       google-play-scraper==1.2.7 \
       huggingface-hub==0.32.3 \
       numba==0.61.2 \
       llvmlite==0.44.0 \
       pynndescent==0.5.13

# Variables de entorno para caches (opcional)
ENV MPLCONFIGDIR=/tmp/matplotlib_config
RUN mkdir -p /tmp/matplotlib_config
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
RUN mkdir -p /tmp/hf_cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -p /tmp/numba_cache

COPY . ${LAMBDA_TASK_ROOT}

CMD ["orchestrator.lambda_handler"]