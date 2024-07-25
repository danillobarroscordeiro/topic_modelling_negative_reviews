import pandas as pd
from awswrangler import athena
import boto3
import json
import logging
import numpy as np
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

with open("variable_hyperparam.json", 'r') as file:
    variable_hyperparam = json.load(file)

sm_client = boto3.client("sagemaker-runtime", region_name=variable_hyperparam['region_name'])


#ENDPOINT_NAME = "topic-modelling-reviews-endpoint-test"

class DatabaseQuery:
    def __init__(
        self, database: str = variable_hyperparam['database'], 
        preprocessed_review_table: str = variable_hyperparam['preprocessed_review_table'],
        products_metadata_table: str = variable_hyperparam['products_metadata'],
        region_name: str = variable_hyperparam['region_name']
):
        self.database = database
        self.preprocessed_review_table = preprocessed_review_table
        self.products_metadata_table = products_metadata_table
        self.session = boto3.Session(region_name=region_name)


    def retrieve_data(self, product_id: str) -> pd.DataFrame:
        query = f"select t1.*, t2.title \
        from {self.database}.{self.preprocessed_review_table} t1 inner join \
        {self.database}.{self.products_metadata_table} t2 on t1.parent_asin = t2.parent_asin  \
        where t1.parent_asin = '{product_id}';"
        df = athena.read_sql_query(sql=query, database=self.database, boto3_session=self.session)
        return df


# def retrieve_data_athena(
#     database=DATABASE, product_id=None, table=PREPROCESSED_REVIEWS_TABLE, boto3_session=SESSION
# ):

#     QUERY= f"select * \
#     from {DATABASE}.{PREPROCESSED_REVIEWS_TABLE} \
#     where parent_asin = '{product_id}';"

#     df = athena.read_sql_query(sql=QUERY, database=database, boto3_session=boto3_session)
#     return df

def calc_mean_embedding(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim == 1:
        return embeddings
    else:
        return np.mean(embeddings, axis=0)


def get_representative_docs(df: pd.Dataframe, column: str = 'embeddings', response: json) -> np.array:
    topic_embeddings = response['topic_embeddings']
    embedding_similarity_lst = []
    for i in teste[column]:
        embedding_similarity = cosine_similarity([i], topic_embeddings)
        embedding_similarity_lst.append(embedding_similarity)

    df['embedding_similarity'] = embedding_similarity_lst
    df = df.sort_values(by='embedding_similarity', ascending=False)
    return df['title_text_review'].tolist()[:3]

def handler(event, context):
    try:

        product_id = json.loads(event['body'])['product_id']
        logger.info(f'product_id: {product_id}')
        
        db_conn = DatabaseQuery()

        df = db_conn.retrieve_data(product_id=product_id)
        product_name = df['title'].values[0]

        embeddings = np.stack(df['embeddings'].values)
        logger.info(f'shape embeddings: {embeddings.shape}')

        mean_embeddings = calc_mean_embedding(embeddings)

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
        output_data = {
            'output_data': {
                'topic_words': response['topic_words'],
                'product_name': product_name,
                'topic_embeddings': representative_docs
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

