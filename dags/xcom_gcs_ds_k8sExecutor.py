from airflow.decorators import task, dag
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook

from datetime import datetime
from kubernetes.client import models as k8s
from io import StringIO

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier


docs = """
By default, Airflow stores all return values in XCom. However, this can introduce complexity, as users then have to consider the size of data they are returning. Futhermore, since XComs are stored in the Airflow database by default, intermediary data is not easily accessible by external systems.
By using an external XCom backend, users can easily push and pull all intermediary data generated in their DAG in GCS.

Additionally using this with the Kubernetes Executor allows more fine grained control over the resources, images, and dependencies for each step on a per task basis.
"""


# Pod Config for the load_data task. This sets resource requests and limits for cpu and memory so that the Pod has enough memory to execute the task.
etl_config={ "pod_override": k8s.V1Pod(
                metadata=k8s.V1ObjectMeta(labels={"purpose": "load_bq_data"}),
                spec=k8s.V1PodSpec(
                    containers=[
                        k8s.V1Container(
                            name="base",
                            resources=k8s.V1ResourceRequirements(
                                limits={
                                    "cpu": 1,
                                    "memory": "8Gi"
                                },
                                requests={
                                    "cpu": 0.5,
                                    "memory": "5Gi"
                                }
                            )
                            )
                        ]
                    )
                )
            }


# Pod Config for train and fit tasks. This sets resource requests and limits for cpu and memory so that the Pod can perform the tasks more quickly. Reduces execution time from minutes to seconds compared to without the use of `KubernetesExecutor`.
modeling_config={ "pod_override": k8s.V1Pod(
                metadata=k8s.V1ObjectMeta(labels={"purpose": "modeling"}),
                spec=k8s.V1PodSpec(
                    containers=[
                        k8s.V1Container(
                            name="base",
                            resources=k8s.V1ResourceRequirements(
                                limits={
                                    "cpu": 2,
                                    "memory": "8Gi"
                                },
                                requests={
                                    "cpu": 1,
                                    "memory": "5Gi"
                                }
                            )
                            )
                        ]
                    )
                )
            }


@dag(
    start_date=datetime(2021, 1, 1),
    schedule_interval=None,
    catchup=False,
    doc_md=docs
)
def using_gcs_for_xcom_ds_k8sExec():

    @task(executor_config=etl_config)
    def load_data():
        """Pull Census data from Public BigQuery and save as Pandas dataframe in GCS bucket with XCom"""

        bq = BigQueryHook()
        sql = """
        SELECT * FROM `bigquery-public-data.ml_datasets.census_adult_income`
        """

        return bq.get_pandas_df(sql=sql, dialect='standard')



    @task
    def preprocessing(df: pd.DataFrame):
        """Clean Data and prepare for feature engineering
        
        Returns pandas dataframe via XCom to GCS bucket.

        Keyword arguments:
        df -- Raw data pulled from BigQuery to be processed. 
        """

        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        # Clean Categorical Variables (strings)
        cols = df.columns
        for col in cols:
            if df.dtypes[col]=='object':
                df[col] =df[col].apply(lambda x: x.rstrip().lstrip())


        # Rename up '?' values as 'Unknown'
        df['workclass'] = df['workclass'].apply(lambda x: 'Unknown' if x == '?' else x)
        df['occupation'] = df['occupation'].apply(lambda x: 'Unknown' if x == '?' else x)
        df['native_country'] = df['native_country'].apply(lambda x: 'Unknown' if x == '?' else x)


        # Drop Extra/Unused Columns
        df.drop(columns=['education_num', 'relationship', 'functional_weight'], inplace=True)

        return df

    @task(executor_config=etl_config)
    def feature_engineering(df: pd.DataFrame):
        """Feature engineering step
        
        Returns pandas dataframe via XCom to GCS bucket.

        Keyword arguments:
        df -- data from previous step pulled from BigQuery to be processed. 
        """

        
        # Onehot encoding 
        df = pd.get_dummies(df, prefix='workclass', columns=['workclass'])
        df = pd.get_dummies(df, prefix='education', columns=['education'])
        df = pd.get_dummies(df, prefix='occupation', columns=['occupation'])
        df = pd.get_dummies(df, prefix='race', columns=['race'])
        df = pd.get_dummies(df, prefix='sex', columns=['sex'])
        df = pd.get_dummies(df, prefix='income_bracket', columns=['income_bracket'])
        df = pd.get_dummies(df, prefix='native_country', columns=['native_country'])


        # Bin Ages
        df['age_bins'] = pd.cut(x=df['age'], bins=[16,29,39,49,59,100], labels=[1, 2, 3, 4, 5])


        # Dependent Variable
        df['never_married'] = df['marital_status'].apply(lambda x: 1 if x == 'Never-married' else 0) 


        # Drop redundant colulmn
        df.drop(columns=['income_bracket_<=50K', 'marital_status', 'age'], inplace=True)

        return {'X': df.drop(columns=['never_married']).to_json(orient='index'), 'y':  df['never_married'].to_json(orient='index')}


    @task(executor_config=modeling_config)
    def cross_validation(df: pd.DataFrame):
        """Train and validate model
        
        Returns accuracy score via XCom to GCS bucket.

        Keyword arguments:
        df -- data from previous step pulled from BigQuery to be processed. 
        """

        y = pd.read_json(StringIO(df['y']), orient='index').values
        X = pd.read_json(StringIO(df['X']), orient='index').values

        model = LGBMClassifier()
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

        return np.mean(n_scores)

    @task(executor_config=modeling_config)
    def fit(accuracy: float, ti=None): 
        """Fit the final model
        
        Determines if accuracy meets predefined threshold to go ahead and fit model on full data set.

        Returns lightgbm model as json via XCom to GCS bucket.
        

        Keyword arguments:
        accuracy -- average accuracy score as determined by CV. 
        """
        if accuracy >= .8:

            # Reuse data produced by the feauture_engineering task by pulling from GCS bucket via XCom
            df = ti.xcom_pull(task_ids='feature_engineering')

            print(f'Training accuracy is {accuracy}. Building Model!')

            y = pd.read_json(StringIO(df['y']), orient='index').values
            X = pd.read_json(StringIO(df['X']), orient='index').values

            model = LGBMClassifier()
            model.fit(X, y)

            return model.booster_.dump_model()

        else:
            return 'Training accuracy ({accuracy}) too low.'


    df = load_data()
    clean_data = preprocessing(df)
    features = feature_engineering(clean_data)
    accuracy = cross_validation(features)
    fit(accuracy)

    # Alternate method to set up task dependencies
    # fit(train(feature_engineering(preprocessing(load_data()))))
    
dag = using_gcs_for_xcom_ds_k8sExec()
