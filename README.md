# Example DAGs for Data Science and Machine Learning Use Cases

These examples are meant to be a guide/skaffold for Data Science and Machine Learning pipelines that can be implemented in Airflow.

In an effort to keep the examples easy to follow, much of the data processing and modeling code has intentially been kept simple.

## Examples

1. `xcom_gcs_ds.py` - A simple DS pipeline from data extraction to modeling.
    - Pulls data from BigQuery using the Google Provider (BigQueryHook) into a dataframe that preps, trains, and builds the model
    - Data is passed between the tasks using [XComs](https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html)
    - Uses GCS as an Xcom backend to easily track intermediary data in a scalable, external system

2. `xcom_gcs_ds_k8sExecutor.py` - A simple DS pipeline from data extraction to modeling that leverages the flexibility of the [Kubernetes Executor](https://www.astronomer.io/blog/new-kubernetesexecutor).
    - All components from example #1 except that each task is now executed in its own pod with custom configs.
    - Uses `pod_override` to provide more resources to tasks to enable proper or faster execution.
