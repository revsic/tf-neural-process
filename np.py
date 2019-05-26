import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import utils
import model

class LatentEncoder:
    def __init__(self, output_sizes, n_latent):
        self.model = utils.dense_sequential(output_sizes)
        self.penultimate = tf.keras.layers.Dense((output_sizes[-1] + n_latent) // 2)
        self.dmu = tf.keras.layers.Dense(n_latent)
        self.dsigma = tf.keras.layers.Dense(n_latent)

    def __call__(self, x, y):
        encoder_input = tf.concat([x, y], axis=-1)
        hidden = self.model(encoder_input)
        hidden = tf.reduce_mean(hidden, axis=1)

        hidden = tf.nn.relu(self.penultimate(hidden))

        mu = self.dmu(hidden)
        log_sigma = self.dsigma(hidden)

        sigma = 0.1 + 0.9 * tf.sigmoid(log_sigma)
        return tfp.distributions.Normal(loc=mu, scale=sigma)


class DeterministicEncoder:
    def __init__(self, output_sizes, attention):
        self.model = utils.dense_sequential(output_sizes)
        self.attention = attention
    
    def __call__(self, cx, cy, tx):
        encoder_input = tf.concat([cx, cy], axis=-1)
        hidden = self.model(encoder_input)
        hidden = self.attention(query=cx, key=tx, value=hidden)
        return hidden


class Decoder:
    def __init__(self, output_sizes):
        self.model = utils.dense_sequential(output_sizes)
    
    def __call__(self, context, tx):
        hidden = tf.concat([context, tx], axis=-1)
        hidden = self.model(hidden)

        mu, log_sigma = tf.split(hidden, 2, axis=-1)
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return dist, mu, sigma


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


class AttentiveNP:
    def __init__(self,
                 latent_encoder_output_sizes,
                 n_latent,
                 decoder_output_sizes,
                 use_deterministic_path=True,
                 deterministic_encoder_output_sizes=None,
                 deterministic_encoder_attention=None):
        self.latent_encoder = LatentEncoder(latent_encoder_output_sizes, n_latent)
        self.decoder = Decoder(decoder_output_sizes)
        self.use_deterministic_path = use_deterministic_path
        if self.use_deterministic_path:
            self.deterministic_encoder = DeterministicEncoder(deterministic_encoder_output_sizes,
                                                              deterministic_encoder_attention)

    def __call__(self, context, query, n_target, target=None):
        q = self.latent_encoder(*context)
        if target is None:
            latent_rep = q.sample()
        else:
            posterior = self.latent_encoder(query, target)
            latent_rep = posterior.sample()

        latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1),
                             [1, n_target, 1])

        if self.use_deterministic_path:
            det_rep = self.deterministic_encoder(*context, query)
            context = tf.concat([det_rep, latent_rep], axis=-1)
        else:
            context = latent_rep

        dist, mu, sigma = self.decoder(context, query)
        if target is not None:
            log_p = dist.log_prob(target)
            kl = tfp.distributions.kl_divergence(q, posterior)
            kl = tf.tile(tf.reduce_sum(kl, axis=-1, keepdims=True),
                         [1, n_target])

            loss = -tf.reduce_sum(log_p - kl / tf.cast(n_target, tf.float32))
        else:
            log_p = None
            kl = None
            loss = None
        
        return mu, sigma, log_p, kl, loss
