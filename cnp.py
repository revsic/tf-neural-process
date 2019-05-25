import tensorflow as tf
import tensorflow_probability as tfp

import utils

class Encoder:
    def __init__(self, output_sizes):
        self.model = utils.dense_sequential(output_sizes)
    
    def __call__(self, context_x, context_y):
        input_tensor = tf.concat([context_x, context_y], axis=-1)
        context = utils.batch_mlp(input_tensor, self.model)
        context = tf.reduce_mean(context, axis=1)
        return context


class Decoder:
    def __init__(self, output_sizes):
        self.model = utils.dense_sequential(output_sizes)
    
    def __call__(self, context, target_x, n_target):
        context = tf.tile(tf.expand_dims(context, axis=1),
                          [1, n_target, 1])
        input_tensor = tf.concat([context, target_x], axis=-1)
        hidden = utils.batch_mlp(input_tensor, self.model)
        
        mu, log_sigma = tf.split(hidden, 2, axis=-1)
        sigma = tf.exp(log_sigma)

        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return dist, mu, sigma


class ConditionalNP:
    def __init__(self, enc_output_sizes, dec_output_sizes):
        self.encoder = Encoder(enc_output_sizes)
        self.decoder = Decoder(dec_output_sizes)

    def __call__(self, context, query, n_target, target=None):
        context = self.encoder(*context)
        dist, mu, sigma = self.decoder(context, query, n_target)

        if target_y is not None:
            log_p = dist.log_prob(target)
        else:
            log_p = None
        
        return log_p, mu, sigma
