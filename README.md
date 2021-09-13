# Example DAGs for Data Science and Machine Learning Use Cases

These examples are meant to be a guide/skaffold for Data Science and Machine Learning pipelines that can be implemented in Airflow.

In an effort to keep the examples easy to follow, much of the data processing and modeling code has intentially been kept simple.

## Examples

1. `xcom_gcs_ds.py` - DS/ML pipeline from data extraction to modeling with Python Operators and GCS Xcom backend.
    - Pulls data from Google BigQuery into a pandas dataframe, prepares data, train and then builds model. 
    - All intermediate data after each task is saved to GCS using Xcom and is also passed between tasks using this method.
    - Model output is saved to GCS via Xcom.
    - All tasks use the python operator.