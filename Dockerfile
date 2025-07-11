# Build an image that can serve mlflow models.
FROM python:3.12.2-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends nginx



WORKDIR /opt/mlflow

# Install MLflow
RUN pip install mlflow==2.19.0

# Copy model to image and install dependencies
COPY model_dir/model /opt/ml/model
RUN python -c "from mlflow.models import container as C; C._install_pyfunc_deps('/opt/ml/model', install_mlflow=False, enable_mlserver=False, env_manager='local');"

ENV MLFLOW_DISABLE_ENV_CREATION=True
ENV ENABLE_MLSERVER=False
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"

# granting read/write access and conditional execution authority to all child directories
# and files to allow for deployment to AWS Sagemaker Serverless Endpoints
# (see https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
RUN chmod o+rwX /opt/mlflow/

# clean up apt cache to reduce image size
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python", "-c", "from mlflow.models import container as C; C._serve('local')"]
