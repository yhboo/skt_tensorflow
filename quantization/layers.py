import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d


class Dense(object):
    """

    """
    def __init__(self, d_in, d_out,
                 use_bias = True, name = ''):
        """
        :param d_in: int, input dimension
        :param d_out: int, output dimension
        :param use_bias: bool, bias flag(if true, bias will be added)
        :param w: numpy(d_in, d_out), if None, w will be random initialized
        :param b: numpy(d_out,)
        :param name: string, layer name
        """
        # exception case controls need to be added.

        #super(Dense, self).__init__(name)
        #keep flags,
        self.use_bias = use_bias

        self.name = name


        #define variables
        with tf.variable_scope(self.name):
            self.w = tf.get_variable('w', (d_in, d_out), dtype=tf.float32, initializer=xavier_initializer())
            if use_bias:
                self.b = tf.get_variable('b', (d_out,), dtype=tf.float32, initializer=tf.zeros_initializer)



    def __call__(self, x):
        if self.use_bias:
            return tf.nn.bias_add(tf.matmul(x, self.w), self.b)
        else:
            return tf.matmul(x, self.w)


class Conv2d(object):
    def __init__(self, d_in, d_out, kernel_size, strides, padding, use_bias = True, name = ''):
        self.d_in = d_in
        self.d_out = d_out
        self.strides = (strides[0], strides[0], strides[1], strides[1])
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.name = name

        #remove string case dependency
        self.padding = padding.upper()

        with tf.variable_scope(self.name):
            self.w = tf.get_variable('kernel', (kernel_size[0], kernel_size[1], d_in, d_out), dtype = tf.float32, initializer=xavier_initializer_conv2d())
            if self.use_bias:
                self.b = tf.get_variable('b', (d_out,), dtype = tf.float32, initializer=tf.zeros_initializer)

    def __call__(self, x):
        if self.use_bias:
            return tf.nn.bias_add(tf.nn.conv2d(x, self.w, self.strides, self.padding), self.b)
        else:
            return tf.nn.conv2d(x, self.w, self.strides, self.padding)

