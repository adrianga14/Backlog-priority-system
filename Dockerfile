FROM public.ecr.aws/lambda/python:3.10

WORKDIR ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt
COPY . ${LAMBDA_TASK_ROOT}

CMD ["orchestrator.lambda_handler"]