import pandas as pd
import os
pd.set_option('display.max_colwidth', None)
import boto3
from awswrangler import config, s3
from sagemaker import get_execution_role, Session
from sagemaker.model import Model
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

image_uri_inference: str = variable_hyperparam['image_uri_inference']


model = Model(
    image_uri=image_uri_inference,
    model_data="s3://sagemaker-us-east-1-513734873949/model-2024-07-10-16-55-21-398/output/model.tar.gz",
    role=role,
    name='inference-model',
    source_dir='../model_deploy_test',
    entry_point='inference.py',
    sagemaker_session=SAGEMAKER_SESSION
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.c5.large",
    endpoint_name="topic-modelling-reviews-endpoint-test"
)