import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
#from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn as biRNN
from tensorflow.contrib.layers import batch_norm, xavier_initializer
from tensorflow.contrib.layers import variance_scaling_initializer

from base_model import quantization_model

class ctc_model(object):
    """
    2 x CONV2 - 4 x 512LSTM(uni) - 30dense
    input shape : (batch, n_frame, 40, 3)
    output shape : (batch, n_frame, 30)
    """
    def __init__(
            self, name = 'ctc',
            kernel_size = (3,3), lstm_dim = 512, n_layer = 3, dropout_ratio = 0.5,
            opt_name = 'Adam', output_dim = 31, mode = 'uni'):
        self.name = name
        self.n_label = output_dim
        self.lstm_dim = lstm_dim
        self.n_layer = n_layer
        self.kernel_size = kernel_size
        self.opt_name = opt_name
        self.mode = mode
        self.dropout_ratio = dropout_ratio



            #self.multiRNNcell_fw = MultiRNNCell(self.lstm_cells_fw)
            #self.multiRNNcell_bw = MultiRNNCell(self.lstm_cells_bw)




    # def set_batch_size(self, new_batch):
    #     self.batch_size = new_batch


    def get_logit(self, input, seq_len, phase):
        keep_prob = tf.cond(phase, lambda:tf.constant(1.0-self.dropout_ratio), lambda:tf.constant(1.0))
        if self.mode == 'uni':
            lstm_cells = []
            for i in range(self.n_layer):
                cell = LSTMCell(self.lstm_dim, initializer = variance_scaling_initializer(1.0, 'FAN_OUT', True))
                cell = DropoutWrapper(cell, output_keep_prob=keep_prob)
                lstm_cells.append(cell)

            multiRNNcell = MultiRNNCell(lstm_cells)

        elif self.mode == 'bi':
            lstm_cells_fw = []
            lstm_cells_bw = []

            for i in range(self.n_layer):
                scope_name = 'lstm_'+str(i)
                with tf.variable_scope(scope_name):
                    cell_fw = LSTMCell(self.lstm_dim/2, initializer = variance_scaling_initializer(1.0, 'FAN_OUT', True))
                    cell_bw = LSTMCell(self.lstm_dim/2, initializer = variance_scaling_initializer(1.0, 'FAN_OUT', True))
                    cell_fw = DropoutWrapper(cell_fw, output_keep_prob = keep_prob)
                    cell_bw = DropoutWrapper(cell_bw, output_keep_prob = keep_prob)
                lstm_cells_fw.append(cell_fw)
                lstm_cells_bw.append(cell_bw)

        # conv1
        with tf.variable_scope(self.name):
            h = tf.layers.conv2d(input, 32, self.kernel_size, (2,2), 'same', use_bias= False, name = 'conv0')
            h = batch_norm(h, center = True, scale = True, is_training=phase, decay=0.99, epsilon=1e-6, scope='bn0')
        h = tf.nn.tanh(h, name='tanh0')

        #conv2
        with tf.variable_scope(self.name):
            h = tf.layers.conv2d(h, 32, self.kernel_size, (2, 2), 'same', use_bias=False, name = 'conv1')
            h = batch_norm(h, center=True, scale=True, is_training=phase, decay=0.99, epsilon=1e-6, scope='bn1')
        h = tf.nn.tanh(h, name='tanh1')

        #reshape
        # ([0] : batch_size, [1] : seq_len, [2]*[3] : feature dimension)
        h_shape = tf.shape(h)

        h = tf.reshape(h, [h_shape[0], h_shape[1], 320])


        #n_layerxLSTM
        if self.mode == 'uni':
            h, _ = tf.nn.dynamic_rnn(
                cell = multiRNNcell,
                inputs = h,
                sequence_length= seq_len,
                dtype = tf.float32,
                scope = self.name
            )
        elif self.mode == 'bi':
            for i in range(self.n_layer):
                scope_name = 'lstm'+str(i)
                h, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw = lstm_cells_fw[i],
                        cell_bw = lstm_cells_bw[i],
                        inputs = h,
                        dtype = tf.float32,
                        scope = scope_name
                        )
                h = tf.concat(h, 2)



        else:
            raise(NotImplementedError)


        h = tf.reshape(h, [-1, self.lstm_dim])

        with tf.variable_scope(self.name+'/dense0'):
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

    def save_all_params(self, sess, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        for pp in tf.global_variables():
            if pp.name.find(self.opt_name) != -1 or pp.name.find('lr') != -1:
                continue
            n_arr = pp.name.split('/')
            n = "_".join(n_arr) + '.npy'
            v = sess.run(pp)
            np.save(path+n, v)
            print(pp.name, ' is saved at ', path+n)



class ctc_model_fixed(quantization_model):
    """
    2 x CONV2 - 4 x 512LSTM(uni) - 30dense
    input shape : (batch, n_frame, 40, 3)
    output shape : (batch, n_frame, 30)
    """

    def __init__(
            self, name='ctc',
            kernel_size = (3,3), lstm_dim = 512, n_layer = 3,
            opt_name = 'Adam', output_dim = 31, dropout_ratio = 0.5,
            n_bits_dict = dict(), pre_trained_path = None
            ):

        super(ctc_model_fixed, self).__init__(n_bits_dict, pre_trained_path)
        self.name = name
        self.n_label = output_dim
        self.lstm_dim = lstm_dim
        self.kernel_size = kernel_size
        self.n_layer = n_layer
        self.lstm_cells = []
        self.opt_name = opt_name

        for i in range(self.n_layer):
            cell = LSTMCell(self.lstm_dim)
            cell = DropoutWrapper(cell, output_keep_prob = (1-dropout_ratio))
            self.lstm_cells.append(cell)

        self.multiRNNcell = MultiRNNCell(self.lstm_cells)

    # def set_batch_size(self, new_batch):
    #     self.batch_size = new_batch


    def get_logit(self, input, seq_len, phase):
        # conv1
        with tf.variable_scope(self.name):
            h = tf.layers.conv2d(input, 32, self.kernel_size, (2, 2), 'same', use_bias=False, name='conv0')
            h = batch_norm(h, center=True, scale=True, is_training=phase, decay=0.99, epsilon=1e-6, scope='bn0')
        h = tf.nn.tanh(h, name='tanh0')

        # conv2
        with tf.variable_scope(self.name):
            h = tf.layers.conv2d(h, 32, self.kernel_size, (2, 2), 'same', use_bias=False, name='conv1')
            h = batch_norm(h, center=True, scale=True, is_training=phase, decay=0.99, epsilon=1e-6, scope='bn1')
        h = tf.nn.tanh(h, name='tanh1')

        # reshape
        # ([0] : batch_size, [1] : seq_len, [2]*[3] : feature dimension)
        h_shape = tf.shape(h)

        h = tf.reshape(h, [h_shape[0], h_shape[1], 320])

        # 4xLSTM
        h, s = tf.nn.dynamic_rnn(
            cell=self.multiRNNcell,
            inputs=h,
            sequence_length=seq_len,
            dtype=tf.float32,
            scope=self.name
        )

        h = tf.reshape(h, [-1, self.lstm_dim])

        with tf.variable_scope(self.name + '/dense0'):
            h = tf.layers.dense(h, self.n_label, kernel_initializer=xavier_initializer())

        h = tf.reshape(h, [h_shape[0], h_shape[1], self.n_label])

        #for pp in tf.global_variables():
        #    self.params.append(pp)
        return h


    def get_ctc_loss(self, logit, seq_len, y):
        loss = tf.nn.ctc_loss(
            inputs=logit,
            labels=y,
            sequence_length=seq_len,
            time_major=False
        )

        cost = tf.reduce_mean(loss)
        predict, _ = tf.nn.ctc_greedy_decoder(tf.transpose(logit, (1, 0, 2)), seq_len)

        cer = tf.reduce_mean(tf.edit_distance(tf.cast(predict[0], tf.int32), y))
        return cost, cer















