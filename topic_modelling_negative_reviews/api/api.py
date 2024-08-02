import gradio as gr
import requests
import json
import logging

logger = logging.getLogger(__name__)

with open('variable_hyperparam.json', 'r') as file:
    variable_hyperparam = json.load(file)


def request_words(product_id: str) -> str:

    input_data = {'product_id': product_id}
    response = requests.post(
        url=variable_hyperparam['lambda_endpoint'], json=input_data
    )
    
    if response.status_code == 200:
        product_name = response.json()['output_data']['product_name']
        topic_words = response.json()['output_data']['topic_words']
        representative_docs = response.json()['output_data']['representative_docs']
        formatted_rep_docs = []
        for i, doc in enumerate(representative_docs, 1):  # Start counting from 1
            formatted_rep_docs.append(f"{i}) {doc}")
        formatted_rep_docs = "\n\n".join(formatted_rep_docs)

        formatted_product_name = f"**Product Name**: {product_name}"
        formmated_topic_words = f"Top words describing negative reviews are: **{topic_words}**"
        formmated_representative_docs = f"**Representative reviews**: \n\n {formatted_rep_docs}" 
        return formatted_product_name, formmated_topic_words, formmated_representative_docs

    else:
        return f"Error: {response.status_code}", response.text

def create_api():
    with gr.Blocks() as api:
        product_id = gr.Textbox(label="Type product ID")
        submit_button = gr.Button("Submit")
        product_name = gr.Markdown()
        topic_words = gr.Markdown()
        formmated_representative_docs = gr.Markdown()
        submit_button.click(
            fn=request_words, 
            inputs=product_id, 
            outputs=[product_name, topic_words, formmated_representative_docs]
            )

    api.launch(server_port=7860, server_name='0.0.0.0')

if __name__ == "__main__":
    create_api()