import tensorflow as tf

# Convert RaggedTensor to a list of lists
def padding_func(sequences):
    sequences = sequences.to_list()
    # Pad sequences to the same length (e.g., maxlen=120)
    padded_sequences = tf.keras.utils.pad_sequences(sequences, maxlen=120, padding='pre', truncating='post') # type: ignore
    padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences)
    # padded_sequences = padded_sequences.batch(padded_sequences.cardinality().numpy())

    return padded_sequences