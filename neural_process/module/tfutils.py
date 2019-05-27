import tensorflow as tf


def dense_sequential(output_sizes, activation=tf.nn.relu):
    """Convert number of hidden units to sequential dense layers
    Args:
        output_sizes: List[int], number of hidden units
        activation: Callable[[tf.Tensor], tf.Tensor], activation function, default ReLU

    Returns:
        tf.keras.Model, sequential model consists of dense layers
    """
    model = tf.keras.Sequential()
    for size in output_sizes[:-1]:
        model.add(tf.keras.layers.Dense(size, activation=activation))
    
    model.add(tf.keras.layers.Dense(output_sizes[-1]))
    return model
