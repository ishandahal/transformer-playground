# transformer-playground

Poking around with transformer based models.

Training transformer from scratch. `src/train_chatGPT.py` file contains single script that trains a replica chatGPT model on data provided by the user. Data is assumed to be a text file. It's a pretty basic script that prints the model output before and after training along with training/validation loss during training.

Coding self-attention mechanism using basic torch `self_attention_from_scratch.ipynb`. Inspiration from following [tutorial by Sebastian Raschka](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
