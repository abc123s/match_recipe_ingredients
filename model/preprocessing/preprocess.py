import json
import os

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
TokenTextEncoder = tfds.deprecated.text.TokenTextEncoder

from preprocessing.tokenizer import IngredientPhraseTokenizer
ingredientPhraseTokenizer = IngredientPhraseTokenizer()


with open(os.path.join(os.path.dirname(__file__), "vocab_list.json")) as vocab_list_data:
    vocab_list = json.load(vocab_list_data)    

word_encoder = TokenTextEncoder(vocab_list,
                                tokenizer=ingredientPhraseTokenizer)

TRUNCATE_LENGTH = 50
# truncate all examples beyond 50 tokens, pad short
# examples up to 50 tokens
def truncate_or_pad(example):
    if len(example) > 50:
        return example[0:50]
    else:
        return example + [0] * (50 - len(example))

# convert triplet training examples into batched tf Dataset
def preprocess_train(examples):
    def example_generator():
        for anchor, positive, negative in examples:
            encoded_anchor = word_encoder.encode(anchor)
            encoded_positive = word_encoder.encode(positive)
            encoded_negative = word_encoder.encode(negative)
            yield (
                truncate_or_pad(encoded_anchor),
                truncate_or_pad(encoded_positive),
                truncate_or_pad(encoded_negative)
            ), 0


    test_dataset = tf.data.Dataset.from_generator(example_generator,
                                                  output_types=((tf.int32, tf.int32, tf.int32), tf.int32))

    return test_dataset.batch(128), word_encoder.vocab_size

# convert batch of manually selected training examples
# into format accepted by model.fit
def preprocess_train_batch(example_batch):
    processed_anchors = []
    processed_positives = []
    processed_negatives = []
    for anchor, positive, negative in example_batch:
        processed_anchors.append(
            truncate_or_pad(word_encoder.encode(anchor))
        )
        processed_positives.append(
            truncate_or_pad(word_encoder.encode(positive))
        )
        processed_negatives.append(
            truncate_or_pad(word_encoder.encode(negative))
        )

    return [np.array(processed_anchors), np.array(processed_positives), np.array(processed_negatives)]

# convert list of raw test examples into format
# that embedding predict on 
def preprocess_test(examples):
    encoded_examples = [truncate_or_pad(word_encoder.encode(example)) for example in examples]

    return tf.constant(encoded_examples)