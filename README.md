# Example DAGs for Data Science and Machine Learning Use Cases

These examples are meant to be a guide/skaffold for Data Science and Machine Learning pipelines that can be implemented in Airflow.

In an effort to keep the examples easy to follow, much of the data processing and modeling code has intentially been kept simple.

## Examples

1. `xcom_gcs_ds.py` - A simple DS pipeline from data extraction to modeling.
    - Pulls data from BigQuery using the Google Provider (BigQueryHook) into a dataframe that preps, trains, and builds the model
    - Data is passed between the tasks using XComs (link to xcom docs)
    - Uses GCS as an XCOM backend to easily track intermediary data in a scalable, external system