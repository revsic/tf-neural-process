import collections
import matplotlib.pyplot as plt
import tensorflow as tf

# Stochastic Process Regression Dataset
SPRDataset = collections.namedtuple(
    'SPRDataset', ('context', 'query', 'target', 'n_context', 'n_target'))

class GPCurvesGenerator:
    """N-dimensional curve generator by Gaussian Process.
    Attributes:
        batch_size: size of batch, number of curves
        max_size: maximum number of generated coordinates
        x_start: start point of x-domain
        x_end: end point of x-domain
        x_dim: dimension of x, number of features in x
        y_dim: dimension of y, number of features in y
        l1_scale: denominator in exponnential term of rbf kernel
        sigma_scale: coefficient of exponential term in rbf kernel
        random_kernel_parameters: randomize kernel parameter
            if true, l1 and sigma value is sampled from uniform dist between 0.1 ~ l1_scale (or sigma_scale)
            if false, l1 and sigma value is diagonal matrix of l1_scale or sigma_scale value
        testing: enable test mode.
            if true, x is np.linspace(x_start, x_end, interval)
            if false, x is sampled from uniform distribution
        test_size: size of tensor on test mode. 
    """
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
        """Generate GP curves
        Returns:
            SPRDataset
                context: (tf.Tensor, tf.Tensor), x-context, y-context, shape=[batch_size, n_context, x_dim (or y_dim)]
                query: tf.Tensor, x-input, shape=[batch_size, n_total, x_dim]
                target: tf.Tensor, y-output, shape=[batch_size, n_total, y_dim]
                n_context: tf.Tensor, number of context
                n_target: tf.Tensor, number of target
        """
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

        return SPRDataset(context=(cx, cy),
                          query=x,
                          target=y,
                          n_context=n_context,
                          n_target=n_total)


def plot_func(x, y, cx, cy, pred, var, batch=0, axis=0):
    """Plot gp function
    Args:
        x: domain of function
        y: function value
        cx: context point, x
        cy: context point, y
        pred: predicted mean
        var: predicted variance
        batch: index of batch to plot
        axis: index of axis to plot
    """
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
