import os
from datetime import datetime
import subprocess
import json

import tensorflow as tf
from tensorflow import keras

from preprocessing.preprocess import preprocess
from model.model import build_model

dataset, vocab_size = preprocess([
[
    'white rice',
    '1 cup white rice, cooked',
    '1 cup brown rice, cooked',
],
[
    'chicken breast',
    '1 pound chicken breast, skinless',
    '3 pounds chicken thigh or chicken drumsticks',
],
[
    'white rice',
    '1 cup white rice, cooked',
    '1 cup brown rice, cooked',
],
[
    'chicken breast',
    '1 pound chicken breast, skinless',
    '3 pounds chicken thigh or chicken drumsticks',
],
[
    'white rice',
    '1 cup white rice, cooked',
    '1 cup brown rice, cooked',
],
[
    'chicken breast',
    '1 pound chicken breast, skinless',
    '3 pounds chicken thigh or chicken drumsticks',
],
[
    'white rice',
    '1 cup white rice, cooked',
    '1 cup brown rice, cooked',
],
[
    'chicken breast',
    '1 pound chicken breast, skinless',
    '3 pounds chicken thigh or chicken drumsticks',
],
[
    'white rice',
    '1 cup white rice, cooked',
    '1 cup brown rice, cooked',
],
[
    'chicken breast',
    '1 pound chicken breast, skinless',
    '3 pounds chicken thigh or chicken drumsticks',
]
])

# hyperparameters
# model structure
WORD_EMBEDDING_SIZE = 128
SENTENCE_EMBEDDING_SIZE = 64
EMBEDDING_ARCHITECTURE = 'simple'

# loss function
TRIPLET_MARGIN = 0.2

# training
OPTIMIZER = 'adam'
EPOCHS = 10

# build model
model, loss = build_model(
    vocab_size = vocab_size,
    word_embedding_size = WORD_EMBEDDING_SIZE,
    sentence_embedding_size = SENTENCE_EMBEDDING_SIZE,
    embedding_architecture = EMBEDDING_ARCHITECTURE,
    triplet_margin = 0.2
)

model.summary()

# make experiment directory and save experiment params down
date_string = datetime.now().strftime("%Y%m%d_%H%M")
commit_string = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
experiment_dir = "experiments/" + date_string + "_" + commit_string
os.mkdir(experiment_dir)

# add tensorboard logs
epoch_tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=experiment_dir + "/epoch_logs", histogram_freq=1)

# compile model
model.compile(loss=loss, optimizer=OPTIMIZER)
history = model.fit(dataset,
                    epochs=EPOCHS,
                    callbacks=[epoch_tensorboard_callback])

# save params down
with open(experiment_dir + "/params.json", "w") as f:
    json.dump(
        {
            "WORD_EMBEDDING_SIZE": WORD_EMBEDDING_SIZE,
            "SENTENCE_EMBEDDING_SIZE": SENTENCE_EMBEDDING_SIZE,
            "EMBEDDING_ARCHITECTURE": EMBEDDING_ARCHITECTURE,
            "TRIPLET_MARGIN": TRIPLET_MARGIN,
            "OPTIMIZER": OPTIMIZER,
            "EPOCHS": EPOCHS,
        },
        f,
        indent=4)

# save model weights for later usage
model.save_weights(experiment_dir + "/model_weights")
