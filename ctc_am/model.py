import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
from tensorflow.contrib.layers import batch_norm, xavier_initializer



class ctc_model(object):
    """
    2 x CONV2 - 4 x 512LSTM(uni) - 30dense
    input shape : (batch, n_frame, 40, 3)
    output shape : (batch, n_frame, 30)
    """
    def __init__(self, name = 'ctc', save_path = './'):
        self.params = None
        self.name = name
        self.n_label = 31
        self.lstm_dim = 512
        self.lstm_cells = []

        self.save_path = save_path

        for i in range(4):
            cell = LSTMCell(self.lstm_dim)
            cell = DropoutWrapper(cell, output_keep_prob=0.5)
            self.lstm_cells.append(cell)

        self.multiRNNcell = MultiRNNCell(self.lstm_cells)


    # def set_batch_size(self, new_batch):
    #     self.batch_size = new_batch


    def get_logit(self, input, seq_len, phase):

        # conv1
        with tf.variable_scope(self.name):
            h = tf.layers.conv2d(input, 32, (3,3), (2,2), 'same', use_bias= False, name = 'conv0')
            h = batch_norm(h, center = True, scale = True, is_training=phase, decay=0.99, epsilon=1e-6, scope='bn0')
        h = tf.nn.tanh(h, name='tanh0')

        #conv2
        with tf.variable_scope(self.name):
            h = tf.layers.conv2d(h, 32, (3, 3), (2, 2), 'same', use_bias=False, name = 'conv1')
            h = batch_norm(h, center=True, scale=True, is_training=phase, decay=0.99, epsilon=1e-6, scope='bn1')
        h = tf.nn.tanh(h, name='tanh1')

        #reshape
        # ([0] : batch_size, [1] : seq_len, [2]*[3] : feature dimension)
        h_shape = tf.shape(h)

        h = tf.reshape(h, [h_shape[0], h_shape[1], 320])


        #4xLSTM
        h, s = tf.nn.dynamic_rnn(
            cell = self.multiRNNcell,
            inputs = h,
            sequence_length= seq_len,
            dtype = tf.float32,
            scope = self.name
        )

        h = tf.reshape(h, [-1, self.lstm_dim])

        with tf.variable_scope('dense0'):
            h = tf.layers.dense(h, self.n_label, kernel_initializer=xavier_initializer())

        h = tf.reshape(h, [h_shape[0], h_shape[1], self.n_label])

        return h



    def get_ctc_loss(self, logit, seq_len, y):
        loss = tf.nn.ctc_loss(
            inputs = logit,
            labels = y,
            sequence_length= seq_len,
            time_major= False
        )

        cost = tf.reduce_mean(loss)
        predict, _ = tf.nn.ctc_greedy_decoder(tf.transpose(logit, (1,0,2)), seq_len)

        cer = tf.reduce_mean(tf.edit_distance(tf.cast(predict[0], tf.int32), y))
        return cost, cer







    # def save_params(self, sess):
    #     param_dict = ()
    #     for pp in self.params:
    #         v = sess.run(pp)
    #         name = pp.name
    #         param_dict[name] = v
    #
    #     with open(self.save_path + self.name+'.pkl', 'wb') as f:
    #         pickle.dump(param_dict, f)
    #
    # def load_params(self, sess):
    #     with open(self.save_path+self.name+'.pkl', 'rb') as f:
    #         param_dict = pickle.load(f)
    #     for pp in self.params:
    #         v = param_dict[pp.name]
    #         sess.run(pp.assign(v))













