FROM python:3.10

# Set the working direction to /api
WORKDIR /api

COPY api/api.py /api

COPY variable_hyperparam.json /api

RUN pip install gradio=="4.37.2"

EXPOSE 7860
CMD ["python", "api.py"]
