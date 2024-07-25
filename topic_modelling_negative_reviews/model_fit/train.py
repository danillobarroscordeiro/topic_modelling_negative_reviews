import pandas as pd
import os
pd.set_option('display.max_colwidth', None)
import boto3
import swifter
from sentence_transformers import SentenceTransformer
from awswrangler import config, s3
from sagemaker import get_execution_role, Session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from textblob import TextBlob
import re
import json

with open('variable_hyperparam.json', 'r') as file:
    variable_hyperparam = json.load(file)

SESSION: str = boto3.Session(
    profile_name=variable_hyperparam['profile_name'], 
    region_name=variable_hyperparam['us-east-1']
)

RUNTIME_CLIENT: str = boto3.client('runtime.sagemaker')

SAGEMAKER_SESSION: str = Session(boto_session=SESSION)

role: str = variable_hyperparam['role']

image_uri_model: str = variable_hyperparam['image_uri_model']

s3_bucket: str = variable_hyperparam['s3_bucket']
SAVE_DATA_S3_PATH: str = f"{s3_bucket}/data/test_model"


df = pd.read_parquet('../../data/interim/df_after_lang_detect.parquet')



df = df[df['language'] != 'ERROR']
df['title_text_review'] = df['title_text_review'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
df['text_length'] = df['title_text_review'].swifter.apply(lambda x: len(x))
df = df[df['text_length'] > 20]
df = df[df['language'] == 'en']
df = df.sample(1000000, random_state=42)
df['review_sentiment'] = df['title_text_review'].swifter.apply(lambda x: TextBlob(x).sentiment.polarity)
df = df[df['review_sentiment'] < -.3]
df.drop(columns=['text_length', 'language', 'review_sentiment', 'parent_asin'], inplace=True)


sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(df['title_text_review'].to_list())

df['embeddings'] = list(embeddings)


s3.to_parquet(
    df=df,
    path=SAVE_DATA_S3_PATH,
    dataset=True,
    filename_prefix="test_model_reviews_",
    mode="overwrite",
    boto3_session=SESSION
)


train_input = TrainingInput(
    "s3://topic-modelling-reviews/data/test_model/",
    content_type="application/x-parquet"
)


estimator = Estimator(
    image_uri=image_uri_model,
    role=role,
    instance_count=1,
    instance_type="ml.c5.2xlarge",
    sagemaker_session=SAGEMAKER_SESSION,
    base_job_name="model"
)

estimator.fit({'training': train_input})