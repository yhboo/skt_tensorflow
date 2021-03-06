import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import time
import tensorflow as tf


from model import ctc_model
from dataset import WSJDataSet
from config import config_fixed
from utils import greedy_decoding


    



def evaluate(cfg):


    charset = cfg['charset']


    #batch size for train & evaluate
    test_batch = cfg['test_batch']


    #paths for data
    base_path = cfg['base_path']
    data_path = cfg['data_path']
    result_path = cfg['result_path']
    model_name = cfg['model_name']
    save_path = 'results/fixed_variables/'

    result_file = result_path + model_name + '.ckpt'


    #load dataset
    dataset = WSJDataSet(test_batch, charset, base_path, data_path = data_path)


    with tf.Graph().as_default():
        # graph construct
        model = ctc_model(model_name)

        #input for the graph
        with tf.variable_scope('PLACEHOLDER'):
            x = tf.placeholder(tf.float32, [None, None, 40, 3], 'x')    # [batch_size, max_seq_len, 40, 3]
#            y = tf.sparse_placeholder(tf.int32, name = 'y')                    #sparse placeholder
            seq_len = tf.placeholder(tf.int32, [None], name = 'seq_len')        # [batch_size]
            phase = tf.placeholder(tf.bool, [], name = 'phase')            #scalar
            new_lr = tf.placeholder(tf.float32, [], name = 'new_lr')

        #connect logit calc graph
        logit= model.get_logit(x, seq_len, phase)

        #define saver
        saver = tf.train.Saver()


        #begin evaluate
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #load parameters
            saver.restore(sess, result_file)


            # evaluation
            dataset.set_batch_size(test_batch)
            dataset.set_mode('test')
            start_time = time.time()
            with tf.name_scope('test'):
                train_phase = False
                while dataset.iter_flag():
                    batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape, label_string = dataset.get_data()
                    cur_logit = sess.run(
                        logit,
                        feed_dict={x: batch_x,
                                   seq_len: batch_seq_len,
                                   #y: (sparse_indices, sparse_values, sparse_shape),
                                   phase: train_phase}
                    )
                    predict = np.argmax(cur_logit, axis = -1)
                    print('Label : ')
                    print(label_string[0][:-1] + ' <\s>')
                    print('predict : ')
                    print(greedy_decoding(predict, charset)+'\n')
            #save model
            for pp in tf.global_variables():
                n_arr = pp.name.split('/')
                p_name = "_".join(n_arr) + '.npy'
                v = sess.run(pp)
                np.save(save_path + p_name, v)


        

if __name__ == '__main__':
    cfg = config_fixed()
    evaluate(cfg)
