import tensorflow as tf
import tensorflow_probability as tfp

import utils

class Encoder:
    def __init__(self, output_sizes, attention=None, keepdims=False):
        self.model = utils.dense_sequential(output_sizes)
        self.attention = attention
        self.keepdims = keepdims

    def __call__(self, rep, key=None, query=None):
        if isinstance(rep, (tuple, list)):
            rep = tf.concat(rep, axis=-1)

        hidden = self.model(rep)
        if self.attention is not None:
            hidden = self.attention(query=query, key=key, value=hidden)
    
        if not self.keepdims:
            hidden = tf.reduce_mean(hidden, axis=1)

        return hidden


class Decoder:
    def __init__(self, output_sizes):
        self.model = utils.dense_sequential(output_sizes)
    
    def __call__(self, context, tx):
        input_tensor = tf.concat([context, tx], axis=-1)
        return self.model(input_tensor)


class GaussianProb:
    def __init__(self, size, multivariate=False, proj=None):
        self.dense_mu = tf.keras.layers.Dense(size)
        self.dense_sigma = tf.keras.layers.Dense(size)
        self.multivariate = multivariate

        self.proj = proj
        if proj is not None:
            self.proj = tf.keras.layers.Dense(proj)
    
    def __call__(self, input_tensor):
        if self.proj is not None:
            input_tensor = self.proj(input_tensor)
        
        mu = self.dense_mu(input_tensor)
        log_sigma = self.dense_sigma(input_tensor)
        
        sigma = tf.exp(log_sigma)
        if self.multivariate:
            dist = tfp.distributions.MultivariateNormalDiag(mu, sigma)
        else:
            dist = tfp.distributions.Normal(mu, sigma)
        
        return dist, mu, sigma
