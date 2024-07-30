import pandas as pd
from awswrangler import athena
import boto3
import json
import logging
import numpy as np
import io
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with open("variable_hyperparam.json", 'r') as file:
    variable_hyperparam = json.load(file)

sm_client = boto3.client("sagemaker-runtime", region_name=variable_hyperparam['aws_region'])


class DatabaseQuery:
    def __init__(
        self, database: str = variable_hyperparam['database'], 
        preprocessed_review_table: str = variable_hyperparam['preprocessed_review_table'],
        products_metadata_table: str = variable_hyperparam['products_metadata_table'],
        region_name: str = variable_hyperparam['aws_region']
):
        self.database = database
        self.preprocessed_review_table = preprocessed_review_table
        self.products_metadata_table = products_metadata_table
        self.session = boto3.Session(region_name=region_name)


    # def retrieve_data(self, product_id: str) -> pd.DataFrame:
    #     query = f"select t1.*, t2.title \
    #     from {self.database}.{self.preprocessed_review_table} t1 inner join \
    #     {self.database}.{self.products_metadata_table} t2 on t1.parent_asin = t2.parent_asin  \
    #     where t1.parent_asin = '{product_id}';"
    #     df = athena.read_sql_query(sql=query, database=self.database, boto3_session=self.session)
    #     return df

    def retrieve_preprocessed_reviews(self, product_id: str) -> pd.DataFrame:
        query = f"select t1.* \
        from {self.database}.{self.preprocessed_review_table} t1 \
        where t1.parent_asin = '{product_id}';"
        df_preprocessed_reviews = athena.read_sql_query(sql=query, database=self.database, boto3_session=self.session)
        return df_preprocessed_reviews

    def retrieve_product_metadata(self, product_id: str) -> pd.DataFrame:
        query = f"select t2.parent_asin, t2.title \
        from {self.database}.{self.products_metadata_table} t2 \
        where t2.parent_asin = '{product_id}';"
        df_product_metadata = athena.read_sql_query(sql=query, database=self.database, boto3_session=self.session)
        return df_product_metadata


def calc_mean_embedding(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.atleast_2d(embeddings)
    if embeddings.shape[0] == 1:
        return embeddings
    else:
        return np.mean(embeddings, axis=0, keepdims=True)

def get_representative_docs(df: pd.DataFrame, column: str = 'embeddings', response: json = {}) -> np.array:
    topic_embeddings = np.array(response['topic_embeddings'])
    logger.info(f"shape topic_embeddings before converting to 2D: {topic_embeddings.shape}")
    topic_embeddings = np.atleast_2d(topic_embeddings)
    logger.info(f"shape topic_embeddings after converting to 2D: {topic_embeddings.shape}")
    embedding_similarity_lst = []
    for i in df[column]:
        embedding = np.atleast_2d(i)
        embedding_similarity = np.max(cosine_similarity(embedding, topic_embeddings))
        embedding_similarity_lst.append(embedding_similarity)

    df['embedding_similarity'] = embedding_similarity_lst
    df = df.sort_values(by='embedding_similarity', ascending=False)
    return df['title_text_review'].tolist()[:3]

def handler(event, context):

    try:
        product_id = json.loads(event['body'])['product_id']
        logger.info(f'product_id: {product_id}')

        db_conn = DatabaseQuery()

        #df = db_conn.retrieve_data(product_id=product_id)
        df_preprocessed_reviews = db_conn.retrieve_preprocessed_reviews(product_id=product_id)
        df_product_metadata = db_conn.retrieve_product_metadata(product_id=product_id)
        logger.info(f"preprocessed dataframe retrieved: {df_preprocessed_reviews.shape} {df_preprocessed_reviews.columns}")
        logger.info(f"metadata dataframe retrieved: {df_product_metadata.shape} {df_product_metadata.columns}")
        df = pd.merge(df_preprocessed_reviews, df_product_metadata, on='parent_asin')
        logger.info(f"merged dataframe: {df.shape} {df.columns}")
        product_name = df['title'].values[0]

        embeddings = np.stack(df['embeddings'].values)
        logger.info(f'shape embeddings: {embeddings.shape}')

        mean_embeddings = calc_mean_embedding(embeddings)
        logger.info(f"mean embeddings done successfully: {mean_embeddings.shape}")

        input_data = io.BytesIO()
        np.save(input_data, mean_embeddings, allow_pickle=True)
        input_data.seek(0)

        response = sm_client.invoke_endpoint(
            EndpointName=variable_hyperparam['inference_endpoint'],
            ContentType="application/x-npy",
            Body=input_data
        )

        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            return {
                'statusCode': response['ResponseMetadata']['HTTPStatusCode'],
                'body': json.dumps({'error': 'Error in calling SageMaker endpoint.'})
            }


        response = json.loads(response['Body'].read().decode())
        representative_docs = get_representative_docs(df=df, response=response)
        logger.info(f"response from the model: f{response}")
        output_data = {
            'output_data': {
                'topic_words': response['topic_words'],
                'product_name': product_name,
                'representative_docs': representative_docs
            }
        }

        logger.info(f'output_data: {output_data}')

        return {
            'statusCode': 200,
            'body': json.dumps(output_data)
        }
    
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Error in decoding JSON'})
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'server error: {str(e)}'})
        }

