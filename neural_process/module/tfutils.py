import tensorflow as tf


def dense_sequential(output_sizes, activation=tf.nn.relu):
    model = tf.keras.Sequential()
    for size in output_sizes[:-1]:
        model.add(tf.keras.layers.Dense(size, activation=activation))
    
    model.add(tf.keras.layers.Dense(output_sizes[-1]))
    return model
