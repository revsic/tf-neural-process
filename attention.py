import tensorflow as tf

import utils

def uniform_attention(query, value):
    total_points = tf.shape(query)[1]
    rep = tf.reduce_mean(value, axis=1, keepdims=True)
    rep = tf.tile(rep, [1, total_points, 1])
    return rep


def laplace_attention(query, key, value, scale, normalize):
    key = tf.expand_dims(key, axis=1)
    query = tf.expand_dims(query, axis=2)
    unnorm_weights = -tf.abs((key - query) / scale)
    unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)
    if normalize:
        weights = tf.nn.softmax(unnorm_weights)
    else:
        weights = 1 + tf.tanh(unnorm_weights)
    return weights @ value


def dot_prod_attention(query, key, value, normalize):
    d_k = tf.shape(key)[-1]
    scale = tf.sqrt(tf.cast(d_k, tf.float32))
    unnorm_weights = key @ tf.transpose(query, [0, 2, 1])
    if normalize:
        weights = tf.nn.softmax(unnorm_weights / scale)
    else:
        weights = tf.nn.sigmoid(unnorm_weights / scale)
    return weights @ value


class MultiheadAttention:
    def __init__(self, n_head):
        self.n_head = n_head
        self.params = None
    
    def __call__(self, query, key, value):
        if self.params is None:
            d_k = key.shape[-1].value
            d_v = value.shape[-1].value
            self.params = self.get_params(d_k, d_v)
        
        rep = tf.constant(0.)
        for param in self.params:
            o = dot_prod_attention(param['dq'](query),
                                   param['dk'](key),
                                   param['dv'](value),
                                   normalize=True)
            rep += param['do'](o)

        return rep

    def get_params(self, d_k, d_v):
        params = []
        head_size = d_v / self.n_head
        for _ in range(self.n_head):
            param = {}
            param['dq'] = tf.keras.layers.Dense(head_size)
            param['dk'] = tf.keras.layers.Dense(head_size)
            param['dv'] = tf.keras.layers.Dense(head_size)
            param['do'] = tf.keras.layers.Dense(d_v)
            params.append(param)
        return params


class Attention:
    def __init__(self, rep, attention_type, scale=1., normalize=True, mlp_output_sizes=None, n_head=8):
        self.rep = rep
        self.attention_type = attention_type
        self.scale = scale
        self.normalize = normalize

        if self.rep not in ['identity', 'mlp']:
            raise ValueError("'rep' not among ['identity', 'mlp']")
        if self.attention_type not in ['uniform', 'laplace', 'dot_product', 'multihead']:
            raise ValueError("'attention_type not among ['uniform', 'laplace', 'dot_product', 'multihead']")

        if self.rep == 'mlp':
            self.dk = utils.dense_sequential(mlp_output_sizes)
            self.dq = utils.dense_sequential(mlp_output_sizes)

        if self.attention_type == 'multihead':
            self.multihead_attention = MultiheadAttention(n_head)

    def __call__(self, key, query, value):
        if self.rep == 'mlp':
            key = utils.batch_mlp(key, self.dk)
            query = utils.batch_mlp(query, self.dq)
        
        if self.attention_type == 'uniform':
            rep = uniform_attention(query, value)
        elif self.attention_type == 'laplace':
            rep = laplace_attention(query, key, value, self.scale, self.normalize)
        elif self.attention_type == 'dot_product':
            rep = dot_prod_attention(query, key, value, self.normalize)
        elif self.attention_type == 'multihead':
            rep = self.multihead_attention(query, key, value)
        
        return rep
