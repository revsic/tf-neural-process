import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from neural_process.module.base import Encoder, Decoder, GaussianProb

class AttentiveNP:
    """Attentive Neural Process
    Attributes:
        z_encoder: Encoder, encoder for latent representation
        z_prob: GaussianProb, latent representation to probability distribution
        encoder: Encoder, context encoder with self attention
        cross_encoder: Encoder, cross context encoder with querying value attention
        decoder: Decoder, decoder for context and latent variable
        normal_dist: GaussianProb, converter for decoded context to probability distribution

    """
    def __init__(self,
                 z_output_sizes,
                 enc_output_sizes,
                 cross_output_sizes,
                 dec_output_sizes,
                 self_attention,
                 cross_attention):
        """Initializer
        Args:
            z_output_sizes: List[int], number of hidden units for latent representation encoder
            enc_output_sizes: List[int], number of hidden units for context encoder
            cross_output_sizes: List[int], number of hidden units for cross context encoder
            dec_output_sizes: List[int], number of hidden units for decoder
            self_attention: Callable[[tf.Tensor], tf.Tensor], self attention method
            cross_attention: Callable[[tf.Tensor], tf.Tensor], cross attention method
        """
        self.z_encoder = Encoder(z_output_sizes[:-1], self_attention)
        self.z_prob = GaussianProb(z_output_sizes[-1],
                                   proj=np.mean(z_output_sizes[-2:]))

        self.encoder = Encoder(enc_output_sizes, self_attention, keepdims=True)
        self.cross_encoder = Encoder(cross_output_sizes, cross_attention)

        self.decoder = Decoder(dec_output_sizes[:-1])
        self.normal_dist = GaussianProb(dec_output_sizes[-1], multivariate=True)

    def __call__(self, context, query):
        cx, _ = context
        z_context = self.z_encoder(context, key=cx, query=cx)
        z_dist, _, _ = self.z_prob(z_context)
        latent = z_dist.sample()

        self_attended = self.encoder(context, key=cx, query=cx)
        cross_attended = self.cross_encoder(self_attended, key=cx, query=query)

        context = tf.concat([latent, cross_attended], axis=-1)
        context = tf.tile(tf.expand_dims(context, 1),
                          [1, tf.shape(query)[1], 1])

        rep = self.decoder(context, query)
        dist, mu, sigma = self.normal_dist(rep)

        return dist, mu, sigma

    def loss(self, context, query, target):
        cx, _ = context
        dist, _, _ = self(context, query)
        log_prob = dist.log_prob(target)
        log_prob = tf.reduce_sum(log_prob)

        prior, _, _ = self.z_prob(self.z_encoder(context, key=cx, query=cx))
        posterior, _, _ = self.z_prob(self.z_encoder([query, target], key=query, query=query))

        kl = tfp.distributions.kl_divergence(prior, posterior)
        kl = tf.reduce_sum(kl)

        # maximize variational lower bound
        loss = -log_prob + kl
        return loss
