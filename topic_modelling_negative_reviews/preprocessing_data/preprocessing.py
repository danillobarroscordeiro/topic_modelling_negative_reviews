import os
pd.set_option('display.max_colwidth', None)
import boto3
from awswrangler import config, s3
from sagemaker import get_execution_role, Session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
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

