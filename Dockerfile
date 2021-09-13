FROM quay.io/astronomer/ap-airflow:2.1.3-1-buster-onbuild

COPY astronomer-cloud-dev-236021-950729c59aa3_workload-id.json .

ENV AIRFLOW__CORE__XCOM_BACKEND=include.gcs_xcom_backend.GCSXComBackend
ENV GOOGLE_APPLICATION_CREDENTIALS=/usr/local/airflow/astronomer-cloud-dev-236021-950729c59aa3_workload-id.json

USER root

RUN apt-get update -y
RUN apt-get install libgomp1 -y

USER astro