from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
import os
import sys
import logging
import argparse
import spacy
import pandas as pd
import numpy as np
import json
from typing import List, Set

with open('variable_hyperparam.json', 'r') as file:
    variable_hyperparam = json.load(file)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


embedding_model = SentenceTransformer(variable_hyperparam['embedding_model'])

def define_nlp(
    max_review_length: int = variable_hyperparam['max_review_length'], 
    disable: List[str] = ['lemmatizer']
):

    nlp = spacy.load("en_core_web_sm", disable=disable)
    nlp.max_length = max_review_length
    return nlp


def spacy_tokenizer(
    doc, 
    token_ent_type: Set[str] = set(variable_hyperparam['token_ent_type']),
    token_dep: Set[str] = set(variable_hyperparam['token_dep']),
    token_head_pos: Set[str] = set(variable_hyperparam['token_head_pos'])
):

    return [
        token.text
        for token in doc 
        if 
        token.is_alpha
        and not token.is_punct
        and not token.is_stop
        and token.ent_type_ not in token_ent_type
        and token.dep_ in token_dep
        and token.head.pos_ in token_head_pos
    ]

def words_to_remove(nlp, words_to_remove: List[str] = variable_hyperparam['words_to_remove']):

    for i in words_to_remove:
        nlp.Defaults.stop_words.add(i)
    
    return None


def define_umap(
    n_neighbors: int = variable_hyperparam['umap_model']['n_neighbors'], 
    n_components: int = variable_hyperparam['umap_model']['n_components'], 
    min_dist: int = variable_hyperparam['umap_model']['min_dist'], 
    metric: str = variable_hyperparam['umap_model']['metric'], 
    random_state: int = variable_hyperparam['umap_model']['random_state']
):

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )

    return umap_model


def define_hdbscan(
    min_cluster_size: int = variable_hyperparam['hdbscan_model']['min_cluster_size'], 
    gen_min_span_tree: bool = variable_hyperparam['hdbscan_model']['gen_min_span_tree'], 
    prediction_data: bool = variable_hyperparam['hdbscan_model']['prediction_data']
):

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        gen_min_span_tree=gen_min_span_tree,
        prediction_data=prediction_data
    )

    return hdbscan_model


def define_vectorizer_tokenizer(
    nlp, ngram_range: tuple = variable_hyperparam['vectorizer_model']['ngram_range'], 
    lowercase: bool = variable_hyperparam['vectorizer_model']['lowercase']
):

    vectorizer_model = CountVectorizer(
        tokenizer= lambda doc: spacy_tokenizer(nlp(doc)),
        ngram_range = ngram_range,
        lowercase= lowercase
    )
    
    return vectorizer_model

def define_representation_model(diversity: float = 0.8):
    representation_model = MaximalMarginalRelevance(diversity=diversity)

    return representation_model

def _save_model(model, model_dir: str):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "my_model")
    
    return model.save(path)

def train(
        df: pd.DataFrame, embedding_model, umap_model, hdbscan_model,
        vectorizer_model, representation_model, model_dir, 
        top_n_words: int = variable_hyperparam['top_n_words'],
        nr_topics: str = 'auto', calculate_probabilities: bool = True,
        language: str = 'english', review_column: str = 'title_text_review',
        embeddings_columns: str = 'embeddings',
        batch_size: int = variable_hyperparam['batch_size'],
        min_similarity: float = variable_hyperparam['min_similarity']
    ):

    topic_models = []
    for i in range(0, len(df), batch_size):
        df_batch = df.iloc[i:i+batch_size]

        topic_model_i = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language=language,
            calculate_probabilities=calculate_probabilities,
            representation_model=representation_model,
            top_n_words=top_n_words,
            nr_topics='auto'
        ).fit(df_batch[review_column], embeddings=np.array(df[embeddings_columns].to_list())[i:i+batch_size])

        topic_models.append(topic_model_i)

    merged_model = BERTopic.merge_models(topic_models, min_similarity=min_similarity)

    # topic_model = BERTopic(
    #     embedding_model=embedding_model,
    #     umap_model=umap_model,
    #     hdbscan_model=hdbscan_model,
    #     vectorizer_model=vectorizer_model,
    #     language=language,
    #     calculate_probabilities=True,
    #     representation_model=representation_model,
    #     top_n_words=top_n_words,
    #     nr_topics=nr_topics
    # )

    # topics, probabilities = topic_model.fit_transform(
    #     df[review_column], embeddings=np.array(df[embeddings_columns].to_list())
    # )
    
    return _save_model(merged_model, model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-data', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--n-neighbors', type=int, default=20)
    parser.add_argument('--n-components', type=int, default=3)
    parser.add_argument('--min-dist', type=float, default=0.1)
    parser.add_argument('--metric', type=str, default='cosine')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--min-cluster-size', type=int, default=70)
    parser.add_argument('--diversity', type=float, default=0.8)
    
    args = parser.parse_args()

    df = pd.read_parquet(args.input_data)

    nlp = define_nlp()
    words_to_remove(nlp=nlp)
    umap_model = define_umap()
    hdbscan_model = define_hdbscan()
    vectorizer_model = define_vectorizer_tokenizer(nlp=nlp)
    representation_model = define_representation_model()
    
    train(
        df, embedding_model=embedding_model, umap_model=umap_model, 
        hdbscan_model=hdbscan_model, model_dir=args.model_dir,
        vectorizer_model=vectorizer_model, representation_model=representation_model
    )


