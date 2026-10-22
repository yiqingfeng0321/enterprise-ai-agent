FROM apache/airflow:3.3.0-python3.11

ARG AIRFLOW_VERSION=3.3.0
ARG PYTHON_VERSION=3.11
ARG AIRFLOW_CONSTRAINTS_URL=https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt

RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    --constraint "${AIRFLOW_CONSTRAINTS_URL}" \
    httpx \
    joblib \
    numpy \
    pandas \
    pydantic \
    python-dotenv \
    PyYAML \
    scikit-learn
