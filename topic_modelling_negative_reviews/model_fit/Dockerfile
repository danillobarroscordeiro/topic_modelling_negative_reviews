FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker

RUN pip install bertopic scikit-learn UMAP HDBSCAN spacy numpy 
RUN python -m spacy download en_core_web_sm


# Set environment variables
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM bert-topic.py

COPY bert-topic.py /opt/ml/code/bert-topic.py
COPY variable_hyperparam.json /opt/ml/code/variable_hyperparam.json

