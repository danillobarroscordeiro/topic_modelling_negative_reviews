from awswrangler import athena, s3
import pandas as pd
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from textblob import TextBlob
import langdetect
import boto3
import swifter
import json
from typing import List


with open("../variable_hyperparam.json", 'r') as file:
    variable_hyperparam = json.load(file)

#SESSION = boto3.Session(profile_name="dbcordeiro_projects", region_name="us-east-1")
# s3_bucket = "s3://topic-modelling-reviews"
# SAVE_DATA_S3_PATH = f"{S3_BUCKET}/data/preprocessed_reviews"
# DATABASE = "amazon_reviews_db"
# RAW_TABLE = "raw_dataset"
# EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
# TABLE = "preprocessed_reviews"
# region_name = 'us-east-1'
# QUERY= f"select title, text, parent_asin \
# from {DATABASE}.{RAW_TABLE} \
# where rating <= 3 \
# and verified_purchase = True;"

#SESSION = boto3.Session(profile_name="dbcordeiro_projects", region_name="us-east-1")
s3_bucket = variable_hyperparam['s3_bucket']
s3_data_preprocessed_path = variable_hyperparam['s3_data_preprocessed_path']
embedding_model = SentenceTransformer(variable_hyperparam['embedding_model'])
preprocessed_review_table = variable_hyperparam['preprocessed_review_table']

class DatabaseQuery:
    def __init__(
        self, 
        database: str = variable_hyperparam['database'], 
        profile_name: str = variable_hyperparam['profile_name'], 
        table: str = variable_hyperparam['raw_table'], 
        region_name: str = variable_hyperparam['aws_region']
    ):
        self.database = database
        self.table = table
        self.session = boto3.Session(profile_name=profile_name, region_name=region_name)

    def retrieve_data(self, rating: int = variable_hyperparam['reviews_rating']) -> pd.DataFrame:
        query = f"select title, text, parent_asin \
        from {self.database}.{self.table} \
        where rating <= {rating} \
        and verified_purchase = True;"
        df = athena.read_sql_query(sql=query, database=self.database, boto3_session=self.session)
        
        return df

# def retrieve_data_athena(database, table, query, boto3_session=None):
#     df = athena.read_sql_query(sql=query, database=database, boto3_session=boto3_session)
#     return df


# def remove_reviews_without_comments(
#     df: pd.DataFrame, columns: List[str] = ['title','text']
# ) -> pd.DataFrame:

#     df_nas = df[
#     (df['title'].isin(['na', 'NA', 'nah', 'NAH', 'n/a', 'N/A', 'None', 'none', 'no', 'NONE', 'NO'])) 
#     & (df['text'].isin(['na', 'NA', 'nah', 'NAH', 'n/a', 'N/A', 'None', 'none', 'no', 'NONE', 'NO']))
#     ]

#     df = df.drop(df_nas.index)

#     return df


def remove_reviews_without_comments(
    df: pd.DataFrame, columns: List[str] = ['title','text']
) -> pd.DataFrame:

    na_values = variable_hyperparam['na_values']
    mask = np.all(df[columns].isin(na_values).values, axis=1)

    filtered_df = df[~mask]

    return filtered_df

# def merge_title_with_review(df):
#     df['title_text_review'] = df[['title', 'text']].swifter.apply(
#     lambda x: x['title'] +  '. ' + x['text'], axis=1
#     )
#     df = df.drop(['title', 'text'], axis=1)
    
#     return df

def merge_title_with_review(
    df: pd.DataFrame, columns: List[str] = ['title','text'], 
    new_column_name: str = 'title_text_review'
) -> pd.DataFrame:

    df[new_column_name] = df[columns].swifter.apply(
    lambda x: x[0] + '. ' + x[1], axis=1
    )
    df = df.drop(columns, axis=1)
    
    return df

def get_review_language(text: str) -> str:

    try:
        return langdetect.detect(text)
    except KeyboardInterrupt as e:
        raise e
    except:
        return 'ERROR'

def apply_language_detection(
    df: pd.DataFrame, column: str = 'title_text_review',
    new_column_name: str = 'language'
) -> pd.DataFrame:

    df[new_column_name] = df[column].swifter.apply(get_review_language)

    return df


def remove_reviews_with_error_language(df: pd.DataFrame, column: str = 'language' ) -> pd.DataFrame:

    df = df[df[column] != 'ERROR']

    return df

def select_language_reviews(
    df: pd.DataFrame, column: str = 'language', language: str = 'en'
) -> pd.DataFrame:

    df = df[df[column] == language]

    return df

def remove_reviews_emoji(
    df: pd.DataFrame, 
    review_column: str = 'title_text_review', 
    regex_mask: str = r"[^\x00-\x7F]+"
) -> pd.DataFrame:
    df[review_column] = df[review_column].apply(lambda x: re.sub(regex_mask, '', x))

    return df

def remove_short_reviews(
    df: pd.DataFrame, 
    min_review_length: int = variable_hyperparam['min_review_length'], 
    review_column: str = 'title_text_review',
    text_length_column: str = 'text_length'
) -> pd.DataFrame:

    df[text_length_column] = df[review_column].swifter.apply(lambda x: len(x))
    df = df[df[text_length_column] > min_review_length]

    return df

def classify_reviews_sentiment(
    df: pd.DataFrame, review_column: str = 'title_text_review', 
    sentiment_column: str = 'review_sentiment'
) -> pd.DataFrame:

    df[sentiment_column] = df[review_column].swifter.apply(lambda x: TextBlob(x).sentiment.polarity)

    return df


def remove_positive_reviews(
    df: pd.DataFrame, 
    polarity: float = variable_hyperparam['polarity'], 
    sentiment_column: str = 'review_sentiment'
) -> pd.DataFrame:

    df = df[df[sentiment_column] < polarity]

    return df

def drop_unecessary_columns(
    df: pd.DataFrame, 
    columns: List[str] = ['text_length', 'language', 'review_sentiment']
) -> pd.DataFrame:

    df.drop(columns=columns, inplace=True)

    return df

def sample_data(df, n=4000000):
    return df.sample(n=n, random_state=42)


def embed_reviews(
    df: pd.DataFrame, review_column: str = 'title_text_review', new_column: str ='embeddings',
    embedding_model: SentenceTransformer = embedding_model
) -> pd.DataFrame:

    reviews = df[review_column]
    reviews_list = reviews.to_list()
    embeddings = embedding_model.encode(reviews_list)
    df[new_column] = list(embeddings)
    
    return df

def normalize_embeddings(
    df: pd.DataFrame, embeddings: str = 'embeddings'
) -> pd.DataFrame:

    norm_embeddings = normalize(df[embeddings].tolist(), 'l2', axis=1)
    df[embeddings] = list(norm_embeddings)
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    df = sample_data(df)
    df = remove_reviews_without_comments(df)
    df = merge_title_with_review(df)
    df = remove_short_reviews(df)
    df = remove_reviews_emoji(df)
    df = classify_reviews_sentiment(df)
    df = remove_positive_reviews(df)
    df = apply_language_detection(df)
    df = remove_reviews_with_error_language(df)
    df = select_language_reviews(df)
    df = drop_unecessary_columns(df)
    df = embed_reviews(df)
    df = normalize_embeddings(df)

    return df


def save_preprocessing_data(
    df: pd.DataFrame, 
    database: str = variable_hyperparam['database'], 
    table: str = variable_hyperparam['preprocessed_review_table'], 
    path: str = variable_hyperparam['s3_data_preprocessed_path'], 
    filename_prefix: str = 'preprocessed_reviews_', 
    mode: str = 'overwrite'
) -> pd.DataFrame:

    s3.to_parquet(
        df=df,
        path=path,
        index=False,
        dataset=True,
        filename_prefix=filename_prefix,
        database=database,
        compression=None,
        table=table,
        mode=mode
    )

    return None

if __name__ == "__main__":
    db_conn = DatabaseQuery(
        database=variable_hyperparam['database'], 
        table=variable_hyperparam['raw_table'], 
        region_name=variable_hyperparam['aws_region']
    )
    df = db_conn.retrieve_data()
    df = preprocess_data(df)
    save_preprocessing_data(df)