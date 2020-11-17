import json
import os

import tensorflow_datasets as tfds

from preprocessing.tokenizer import IngredientPhraseTokenizer
from model.model import build_model
from evaluate import evaluate

ingredientPhraseTokenizer = IngredientPhraseTokenizer()
TokenTextEncoder = tfds.deprecated.text.TokenTextEncoder

with open(os.path.join(os.path.dirname(__file__), "preprocessing/vocab_list.json")) as vocab_list_data:
    vocab_list = json.load(vocab_list_data)    

word_encoder = TokenTextEncoder(vocab_list,
                                tokenizer=ingredientPhraseTokenizer)

experiment_dir = "experiments/20201117_0333_f605d42"

# load experiment params
with open(experiment_dir + "/params.json", "r") as f:
    params = json.load(f)

# build and compile model based on experiment params:
model, _ = build_model(
    vocab_size = word_encoder.vocab_size,
    word_embedding_size = params["WORD_EMBEDDING_SIZE"],
    sentence_embedding_size = params["SENTENCE_EMBEDDING_SIZE"],
    embedding_architecture = params["EMBEDDING_ARCHITECTURE"],
    triplet_margin = params["TRIPLET_MARGIN"],
)

# load final weights from experiment into model:
model.load_weights(experiment_dir + "/model_weights")

# evaluate model and save metrics:
evaluation = evaluate(model)

with open(experiment_dir + "/results.json", "w") as f:
    json.dump(evaluation, f, indent=4)
