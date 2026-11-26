# Airflow image with the pipeline's dependencies; the code, DAG and models are mounted by docker-compose.
FROM apache/airflow:2.10.3-python3.11

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
