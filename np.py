import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import model

class NeuralProcess:
    def __init__(self,
                 z_output_sizes,
                 enc_output_sizes,
                 dec_output_sizes):
        self.z_encoder = model.Encoder(z_output_sizes[:-1])
        self.z_prob = model.GaussianProb(z_output_sizes[-1],
                                         proj=np.mean(z_output_sizes[-2:]))

        self.encoder = model.Encoder(enc_output_sizes)
        self.decoder = model.Decoder(dec_output_sizes[:-1])
        self.normal_dist = model.GaussianProb(dec_output_sizes[-1], multivariate=True)
    
    def __call__(self, context, query):
        z_context = self.z_encoder(*context)
        z_dist, _, _ = self.z_prob(z_context)

        latent = z_dist.sample()
        context = self.encoder(*context)

        context = tf.concat([latent, context], axis=-1)
        context = tf.tile(tf.expand_dims(context, 1),
                          [1, tf.shape(query)[1], 1])
        
        rep = self.decoder(context, query)
        dist, mu, sigma = self.normal_dist(rep)

        return dist, mu, sigma
    
    def loss(self, context, query, target):
        dist, _, _ = self(context, query)
        log_prob = dist.log_prob(target)
        log_prob = tf.reduce_sum(log_prob)

        prior, _, _ = self.z_prob(self.z_encoder(*context))
        posterior, _, _ = self.z_prob(self.z_encoder(query, target))

        kl = tfp.distributions.kl_divergence(prior, posterior)
        kl = tf.reduce_sum(kl)

        loss = -log_prob + kl
        return loss
