import matplotlib.pyplot as plt
import tensorflow as tf


def dense_sequential(output_sizes, activation=tf.nn.relu):
    model = tf.keras.Sequential()
    for size in output_sizes[:-1]:
        model.add(tf.keras.layers.Dense(size, activation=activation))
    
    model.add(tf.keras.layers.Dense(output_sizes[-1]))
    return model


def plot_func(x, y, cx, cy, pred, var, batch=0, axis=0):
    plt.plot(x[batch], pred[batch], 'b', linewidth=2)
    plt.plot(x[batch], y[batch], 'k:', linewidth=2)
    plt.plot(cx[batch], cy[batch], 'ko', markersize=10)
    plt.fill_between(
        x[batch, :, axis], 
        pred[batch, :, axis] - var[batch, :, axis],
        pred[batch, :, axis] + var[batch, :, axis],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    plt.show()
