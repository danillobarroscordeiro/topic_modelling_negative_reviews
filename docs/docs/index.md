# topic_modelling_negative_reviews documentation!

## Description

This project aims to summarize all negative reviews of a product and classify them in one of the topics in a few words to detect which are the main customer complaints about it. The model is deployed in AWS and stakeholders can access these words by entering the product ID in a gradio web API.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://s3://topic-modelling-reviews/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://s3://topic-modelling-reviews/data/` to `data/`.


