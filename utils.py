import matplotlib.pyplot as plt
import tensorflow as tf


def dense_sequential(output_sizes, activation=tf.nn.relu):
    model = tf.keras.Sequential()
    for size in output_sizes[:-1]:
        model.add(tf.keras.layers.Dense(size, activation=activation))
    
    model.add(tf.keras.layers.Dense(output_sizes[-1]))
    return model


def batch_mlp(input_tensor, model):
    batch_size, _, filter_size = input_tensor.shape.as_list()
    output = tf.reshape(input_tensor, (-1, filter_size))

    output = model(output)
    output_size = output.shape[-1].value

    output = tf.reshape(output, (batch_size, -1, output_size))
    return output


# def plot_func(x, y, cx, cy, )