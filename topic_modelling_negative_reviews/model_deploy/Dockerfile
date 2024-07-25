FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker

ENV PATH="/opt/ml/code:${PATH}"

COPY inference.py /opt/ml/code/inference.py

COPY variable_hyperparam.json /opt/ml/code/variable_hyperparam.json

# copy bert pretrained model
COPY all-MiniLM-L6-v2 /opt/ml/code/sentence_transformers/all-MiniLM-L6-v2

RUN pip install bertopic scikit-learn UMAP HDBSCAN numpy 
RUN pip install -U sentence-transformers

# Set environment variables
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM inference.py


