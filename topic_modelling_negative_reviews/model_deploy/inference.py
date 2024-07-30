from bertopic import BERTopic 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import argparse
import io
import logging
import json

with open('/opt/ml/code/variable_hyperparam.json', 'r') as file:
    variable_hyperparam = json.load(file)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_fn(model_dir):

    try:
        logger.info(f"inside model_fn, model_dir: {model_dir}")
        logger.info(f"check model_dir: {os.listdir(model_dir)}")
        # Load the SentenceTransformer model from the local directory
        embedding_model_path = variable_hyperparam['embedding_model_path']

        if not os.path.exists(embedding_model_path):
            raise ValueError(f"Model path does not exist: {embedding_model_path}")

        embedding_model = SentenceTransformer(embedding_model_path)
        model = BERTopic.load(model_dir)
        model.embedding_model = embedding_model
        topic_embeddings = model.topic_embeddings_
        logger.info("Model loaded successfully.")

        return model, topic_embeddings

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

def input_fn(input_data, content_type):

    try:
        if content_type == 'application/x-npy':
            buffer = io.BytesIO(input_data)
            embeddings = np.load(buffer, allow_pickle=True)
            logger.info(f"embeddings processed successfully: {embeddings}")
            logger.info(f"embeddings shape: {embeddings.shape}")

            return {'embeddings': embeddings}

    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")

def predict_closest_topic(first_embeddings, topic_embeddings):

    try:
        logger.info(f"embeddings shape: {first_embeddings.shape}")
        similarities = cosine_similarity(first_embeddings, topic_embeddings)
        predicted_topic = np.argmax(similarities)
        print(predicted_topic)

        return predicted_topic

    except Exception as e:
        logger.error(f"Error in predict_closest_topic: {str(e)}")


def get_words_from_topics(predicted_topic, topic_model):

    try:
        top_words = []
        topic_words_probs = topic_model.topic_representations_[predicted_topic]

        for i in topic_words_probs:
            top_words.append(i[0])

        top_words_str = " ".join(top_words)
        logger.info(f'top words: {top_words_str}')

        return top_words_str

    except Exception as e:
        logger.error(f"Error in get_words_from_topics: {str(e)}")

def predict_fn(data, model):

    try:
        model, topic_embeddings = model
        embeddings = data['embeddings']

        predicted_topic = predict_closest_topic(embeddings, topic_embeddings)
        topic_words = get_words_from_topics(topic_model=model, predicted_topic=predicted_topic)
        topic_embeddings = model.topic_embeddings_[predicted_topic].tolist()
        prediction = {'topic_words': topic_words, 'topic_embeddings': topic_embeddings}
        logger.info(f'prediction value: {topic_words}')
        logger.info(f'topic_embedding value: {topic_embeddings}')

        return prediction

    except Exception as e:

        logger.error(f"Error in predict_fn: {str(e)}")



def output_fn(prediction, accept):

    try:
        if accept == 'application/json':
            return json.dumps(prediction)
        
        else:
            raise ValueError("Unsupported accept type: {}".format(accept))

    except Exception as e:
        logger.error(f"Error in output_fn: {str(e)}")