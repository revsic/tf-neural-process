import tensorflow as tf

from neural_process.module.tfutils import dense_sequential

def uniform_attention(query, _, value):
    # query : [B, T, X]
    # value : [B, C, X]
    n_target = tf.shape(query)[1]
    rep = tf.reduce_mean(value, axis=1, keepdims=True)
    rep = tf.tile(rep, [1, n_target, 1])
    # [B, T, X]
    return rep


def laplace_attention(query, key, value):
    # query : [B, T, X]
    # key : [B, C, X]
    # value : [B, C, X]
    key = tf.expand_dims(key, axis=1)
    query = tf.expand_dims(query, axis=2)
    # unnorm_weights : [B, T, C, X]
    unnorm_weights = -tf.abs((key - query))
    # unnorm_weights : [B, T, C]
    unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)
    weights = tf.nn.softmax(unnorm_weights)
    # [B, T, X]
    return weights @ value


def dotprod_attention(query, key, value):
    # query : [B, T, X]
    # key : [B, C, X]
    # value : [B, C, X]
    d_k = tf.shape(key)[-1]
    scale = tf.sqrt(tf.cast(d_k, tf.float32))
    # unnorm_weights : [B, T, C]
    unnorm_weights = query @ tf.transpose(key, [0, 2, 1])
    weights = tf.nn.softmax(unnorm_weights / scale)
    # [B, T, X]
    return weights @ value


class MultiheadAttention:
    def __init__(self, n_head):
        self.n_head = n_head
        self.params = None
    
    def __call__(self, query, key, value):
        if self.params is None:
            d_v = value.shape[-1].value
            self.params = self.get_params(d_v)

        rep = tf.constant(0.)
        for param in self.params:
            o = dotprod_attention(param['dq'](query),
                                   param['dk'](key),
                                   param['dv'](value))
            rep += param['do'](o)

        return rep

    def get_params(self, d_v):
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
    """Wrapper for attention mechanisms
    Attributes:
        attention_type: str, type of attention, [uniform, laplace, dotprod, multihead]
        proj: List[int], number of hidden units for projection layer
            if None, projection is not applied
        dk: optional, Callable[[tf.Tensor], tf.Tensor], dense sequential for key projection
        dq: optional, Callable[[tf.Tensor], tf.Tensor], dense sequential for query projection
        multihead_attention, optional, MultiheadAttention, object for multihead attention
    """
    def __init__(self, attention_type, proj=None, n_head=8):
        self.attention_type = attention_type
        if self.attention_type not in ['uniform', 'laplace', 'dotprod', 'multihead']:
            raise ValueError("'attention_type should be one of ['uniform', 'laplace', 'dotprod', 'multihead']")

        self.proj = proj
        if self.proj is not None:
            self.dk = dense_sequential(self.proj)
            self.dq = dense_sequential(self.proj)

        if self.attention_type == 'multihead':
            self.multihead_attention = MultiheadAttention(n_head)

    def __call__(self, query, key, value):
        if self.proj is not None:
            key = self.dk(key)
            query = self.dq(query)

        rep = self.attention_fn(query, key, value)
        return rep

    def attention_fn(self, query, key, value):
        attentions = {
            'uniform': uniform_attention,
            'laplace': laplace_attention,
            'dotprod': dotprod_attention
        }
        if self.attention_type != 'multihead':
            attn = attentions[self.attention_type]
        else:
            attn = self.multihead_attention
        return attn(query, key, value)
