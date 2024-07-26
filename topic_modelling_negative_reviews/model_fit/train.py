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

with open('../variable_hyperparam.json', 'r') as file:
    variable_hyperparam = json.load(file)

SESSION: str = boto3.Session(
    profile_name=variable_hyperparam['profile_name'], 
    region_name=variable_hyperparam['aws_region']
)

RUNTIME_CLIENT: str = boto3.client('runtime.sagemaker')

SAGEMAKER_SESSION: str = Session(boto_session=SESSION)

role: str = variable_hyperparam['role']

image_uri_model: str = variable_hyperparam['image_uri_model']

s3_bucket: str = variable_hyperparam['s3_bucket']
SAVE_DATA_S3_PATH: str = f"{s3_bucket}/data/test_model"



train_input = TrainingInput(
    "s3://topic-modelling-reviews/data/preprocessed_reviews/",
    content_type="application/x-parquet"
)


estimator = Estimator(
    image_uri=image_uri_model,
    role=role,
    instance_count=1,
    instance_type="ml.c5.4xlarge",
    sagemaker_session=SAGEMAKER_SESSION,
    base_job_name="model"
)

estimator.fit({'training': train_input})