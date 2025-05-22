FROM public.ecr.aws/lambda/python:3.10

WORKDIR ${LAMBDA_TASK_ROOT}
COPY . ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir \
    boto3 pandas numpy gensim stop-words joblib scikit-learn google_play_scraper \
    -t ${LAMBDA_TASK_ROOT}

CMD ["orchestrator.lambda_handler"]