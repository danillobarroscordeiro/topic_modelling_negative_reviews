import gradio as gr
import requests
import json

with open('variable_hyperparam.json', 'r') as file:
    variable_hyperparam = json.load(file)

#endpoint = "https://dk5nrp46w2af4blzrctvyuqvfe0lgmhz.lambda-url.us-east-1.on.aws/"

def request_words(product_id: str) -> str:

    input_data = {'product_id': product_id}
    response = requests.post(
        url=variable_hyperparam['lambda_endpoint'], json=input_data
    )
    
    if response.status_code == 200:
        cluster_words = response.json()['output_data']['cluster_words']
        product_name = response.json()['output_data']['product_name']

        return cluster_words, product_name

    else:
        return f"Error: {response.status_code}", response.text

def create_api():
    with gr.Blocks() as api:
        product_id = gr.Textbox(label="Type product ID")
        submit_button = gr.Button("Submit")
        product_name = gr.Markdown(label="Product Name:")
        cluster_words = gr.Markdown(label="The top words that describes the negatives reviews are:")
        submit_button.click(fn=request_words, inputs=product_id, outputs=[product_name, cluster_words])

    api.launch(server_port=7860, server_name='0.0.0.0')

if __name__ == "__main__":
    create_api()