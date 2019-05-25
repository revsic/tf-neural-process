import collections
import tensorflow as tf

# Stochastic Process Regression Dataset
SPRDataset = collections.namedtuple(
    'SPRDataset', ('context', 'query', 'target', 'n_context', 'n_target'))

class GPCurvesGenerator:
    def __init__(self,
                 batch_size,
                 max_size,
                 x_start=-2,
                 x_end=2,
                 x_dim=1,
                 y_dim=1,
                 l1_scale=0.6,
                 sigma_scale=1.,
                 random_kernel_parameters=True,
                 testing=False,
                 test_size=400):
        self.batch_size = batch_size
        self.max_size = max_size
        self.x_start = x_start
        self.x_end = x_end
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.l1_scale = l1_scale
        self.sigma_scale = sigma_scale
        self.random_kernel_parameters = random_kernel_parameters
        self.testing = testing
        self.test_size = test_size
    
    def gaussian_kernels(self, xdata, l1, sigma, sigma_noise=2e-2):
        n_context = tf.shape(xdata)[1]

        # xdata : [B, N, X]
        # diff : [B, N, N, X]
        diff = xdata[:, None, :, :] - xdata[:, :, None, :]
        # l1 : [B, Y, X]
        # dist : [B, Y, N, N, X]
        dist = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])
        # dist : [B, Y, N, N]
        dist = tf.reduce_sum(dist, axis=-1)

        # sigma : [B, Y]
        # kernel : [B, Y, N, N]
        kernel = tf.square(sigma)[:, :, None, None] * tf.exp(-1. / 2 * dist)
        kernel += (sigma_noise ** 2) * tf.eye(n_context)

        return kernel
    
    def generate(self):
        n_context = tf.random_uniform([], 3, self.max_size, dtype=tf.int32)

        if self.testing:
            if self.x_dim != 1:
                raise ValueError('Test mode is only available at x-dim == 1')
            
            if self.test_size < self.max_size:
                raise ValueError('test_size must be bigger than max_size')

            n_target = self.test_size
            n_total = n_target
            interval = float(self.x_end - self.x_start) / n_target
            # x : [B, self.test_size]
            x = tf.tile(tf.range(self.x_start, self.x_end, interval, dtype=tf.float32)[None, :],
                        [self.batch_size, 1])
            # x : [B, self.test_size, 1]
            x = tf.expand_dims(x, axis=-1)
        else:
            n_target = tf.random_uniform([], 3, self.max_size, dtype=tf.int32)
            n_total = n_target + n_context
            # x : [B, n_total, self.x_dim]
            x = tf.random_uniform([self.batch_size, n_total, self.x_dim], self.x_start, self.x_end)
        
        param_shape = [self.batch_size, self.y_dim, self.x_dim]
        if self.random_kernel_parameters:
            l1 = tf.random_uniform(param_shape, 0.1, self.l1_scale)
            sigma = tf.random_uniform(param_shape[:-1], 0.1, self.sigma_scale)
        else:
            l1 = tf.ones(param_shape) * self.l1_scale
            sigma = tf.ones(param_shape[:-1]) * self.sigma_scale
        
        kernel = self.gaussian_kernels(x, l1, sigma)
        cholesky = tf.cholesky(kernel)

        y = cholesky @ tf.random_normal([self.batch_size, self.y_dim, n_total, 1])
        y = tf.transpose(tf.squeeze(y, 3), [0, 2, 1])

        if self.testing:
            idx = tf.random_shuffle(tf.range(n_total))
            cx = tf.gather(x, idx[:n_context], axis=1)
            cy = tf.gather(y, idx[:n_context], axis=1)
        else:
            cx = x[:, :n_context, :]
            cy = y[:, :n_context, :]
        
        context = (cx, cy)
        query = x
        target = y
        return SPRDataset(context=(cx, cy),
                          query=x,
                          target=y,
                          n_context=n_context,
                          n_target=n_total)
