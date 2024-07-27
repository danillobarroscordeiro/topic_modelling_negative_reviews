FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker

RUN pip install bertopic=="0.16.2" scikit-learn=="1.5.0" umap-learn=="0.5.6" HDBSCAN=="0.8.37" spacy=="3.7.5" numpy 
RUN python -m spacy download en_core_web_sm


# Set environment variables
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM bert-topic.py

COPY model_fit/bert-topic.py /opt/ml/code/bert-topic.py
COPY variable_hyperparam.json /opt/ml/code/variable_hyperparam.json

