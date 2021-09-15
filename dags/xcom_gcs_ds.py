from airflow.decorators import task, dag
from airflow.models import DAG
from airflow.utils.dates import days_ago

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier

from google.cloud import bigquery


@dag(
    default_args={'owner': 'airflow'},
    start_date=days_ago(2),
    schedule_interval=None,
    catchup=False
)
def using_gcs_for_xcom_ds():

    @task
    def load_data():
        """Pull Census data from Public BigQuery and save as Pandas dataframe in GCS bucket with XCom"""

        client = bigquery.Client()
        sql = """
        SELECT * FROM `bigquery-public-data.ml_datasets.census_adult_income`
        """
        raw_data = client.query(sql).to_dataframe()
        return raw_data


    @task
    def preprocessing(df: pd.DataFrame):
        """Clean Data and prepare for feature engineering
        
        Returns pandas dataframe via Xcom to GCS bucket.

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

    @task
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

        return df


    @task
    def train(df: pd.DataFrame):
        """Train and validate model
        
        Returns accuracy score via XCom to GCS bucket.

        Keyword arguments:
        df -- data from previous step pulled from BigQuery to be processed. 
        """

        y = df['never_married'].values
        X = df.drop(columns=['never_married']).values


        model = LGBMClassifier()
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

        return np.mean(n_scores)

    @task
    def fit(accuracy: float, **kwargs): 
        """Fit the final model
        
        Determines if accuracy meets predefined threshold to go ahead and fit model on full data set.

        Returns lightgbm model as json via XCom to GCS bucket.
        

        Keyword arguments:
        accuracy -- average accuracy score as determined by CV. 
        """
        if accuracy >= .8:

            # Reuse data produced by the feauture_engineering task by pulling from GCS bucket via XCom
            df = kwargs['ti'].xcom_pull(task_ids='feature_engineering')

            print(f'Training accuracy is {accuracy}. Building Model!')
            y = df['never_married'].values
            X = df.drop(columns=['never_married']).values


            model = LGBMClassifier()
            model.fit(X, y)

            return model.booster_.dump_model()

        else:
            return 'Training accuracy ({accuracy}) too low.'


    df = load_data()
    clean_data = preprocessing(df)
    features = feature_engineering(clean_data)
    accuracy = train(features)
    fit(accuracy)

    # Alternate method to set up task dependencies
    # fit(train(feature_engineering(preprocessing(load_data()))))
    
dag = using_gcs_for_xcom_ds()
