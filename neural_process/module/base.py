import tensorflow as tf
import tensorflow_probability as tfp

from neural_process.module.tfutils import dense_sequential

class Encoder:
    """Context encoder
    Attributes:
        model: Callable[[tf.Tensor], tf.Tensor], dense sequential encoder
        attention: Callable[[tf.Tensor], tf.Tensor], attention method, default None
            if None, attention is not applied
        keepdims: bool, if false, reduce axis 1 with mean method
    """
    def __init__(self, output_sizes, attention=None, keepdims=False):
        self.model = dense_sequential(output_sizes)
        self.attention = attention
        self.keepdims = keepdims

    def __call__(self, rep, key=None, query=None):
        """Encode given context
        Args:
            rep: tf.Tensor or tuple, list of tf.Tensor, representation
                if rep consists of multiple tensor, concatenate it to decode
            key: tf.Tensor, key for attention method, default None
            query: tf.Tensor, query for attention method, default None

        Returns:
            tf.Tensor, encoded context
        """
        if isinstance(rep, (tuple, list)):
            rep = tf.concat(rep, axis=-1)

        hidden = self.model(rep)
        if self.attention is not None:
            hidden = self.attention(query=query, key=key, value=hidden)
    
        if not self.keepdims:
            hidden = tf.reduce_mean(hidden, axis=1)

        return hidden


class Decoder:
    """Context decoder
    Attributes:
        model: Callable[[tf.Tensor], tf.Tensor], dense sequential decoder
    """
    def __init__(self, output_sizes):
        self.model = dense_sequential(output_sizes)
    
    def __call__(self, context, tx):
        """Decode tensor
        Args:
            context: tf.Tensor, encoded context
            tx: tf.Tesnor, query
        
        Returns:
            tf.Tensor, decoded value
        """
        input_tensor = tf.concat([context, tx], axis=-1)
        return self.model(input_tensor)


class GaussianProb:
    """Convert input tensor to gaussian distribution representation
    Attributs:
        dense_mu: Callable[[tf.Tensor], tf.Tensor], dense layer for mean
        dense_sigma: Callable[[tf.Tensor], tf.Tensor], dense layer for sigma
        multivariate: bool, if true, return multivariate gaussian distribution with diagonal covariance
        proj: Callable[[tf.Tensor], tf.Tensor], projection layer for input tensor, default None
            if None, projection is not applied
    """
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
