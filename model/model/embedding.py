from tensorflow import keras

def build_embedding(
    vocab_size,
    word_embedding_size,
    sentence_embedding_size,
    architecture,
):
    if architecture == 'simple':
        return keras.Sequential(
            [
                keras.layers.Embedding(vocab_size, word_embedding_size, mask_zero = True),
                keras.layers.LSTM(sentence_embedding_size)
            ],
            name = 'embedding'
        )
    if architecture == 'bidirectional':
        return keras.Sequential(
            [
                keras.layers.Embedding(vocab_size, word_embedding_size, mask_zero = True),
                keras.layers.Bidirectional(
                    keras.layers.LSTM(sentence_embedding_size),
                    merge_mode = 'ave'
                ),
            ],
            name = 'embedding'
        )
