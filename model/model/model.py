import tensorflow as tf
from tensorflow import keras

from model.embedding import build_embedding 

def build_model(
    vocab_size,
    word_embedding_size,
    sentence_embedding_size,
    embedding_architecture,
    triplet_margin
):
    # construct model
    embedding = build_embedding(
        vocab_size = vocab_size,
        word_embedding_size = word_embedding_size,
        sentence_embedding_size = sentence_embedding_size,
        architecture = embedding_architecture,
    )

    anchor = keras.layers.Input(shape=(50,), name = 'anchor')
    positive = keras.layers.Input(shape=(50,), name = 'positive')
    negative = keras.layers.Input(shape=(50,), name = 'negative')

    anchor_embedding = embedding(anchor)
    positive_embedding = embedding(positive)
    negative_embedding = embedding(negative)

    output = keras.layers.concatenate(
        [anchor_embedding, positive_embedding, negative_embedding],
        axis = 1
    )

    model = keras.models.Model([anchor, positive, negative], output)

    # construct triplet loss function
    def triplet_loss(_, y_pred):
        # extract embeddings for anchor, positive, and negative examples
        anchor_embeddings = y_pred[:,:sentence_embedding_size]
        positive_embeddings = y_pred[:,sentence_embedding_size:2*sentence_embedding_size]
        negative_embeddings = y_pred[:,2*sentence_embedding_size:]

        # compute (mean) distance between anchors and positives and anchors and negatives
        positive_dist = tf.math.reduce_mean(tf.square(anchor_embeddings - positive_embeddings), axis=1)
        negative_dist = tf.math.reduce_mean(tf.square(anchor_embeddings - negative_embeddings), axis=1)

        # compute triplet loss with specified margin
        return tf.math.maximum(positive_dist - negative_dist + triplet_margin, 0.)

    return model, triplet_loss


