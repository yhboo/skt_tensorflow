import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import time
import tensorflow as tf
import pickle


from model import ctc_model_fixed
from dataset import WSJDataSet
from config import config_fixed
from utils import greedy_decoding, WER

def train(cfg):


    charset = cfg['charset']
    am_output_dim = len(charset)+1
    
    #model parameters
    opt_name = cfg['opt_name']
    kernel_size = cfg['kernel_size']
    lstm_dim = cfg['lstm_dim']
    lstm_n_layer = cfg['n_layer']
    dr_ratio = cfg['dropout_ratio']

    #for early stopping technique
    initial_lr = cfg['initial_lr']
    decay_factor = cfg['decay_factor']
    decay_time = cfg['decay_time']
    patience = cfg['patience']
    max_epoch = cfg['max_epoch']

    #batch size for train & evaluate
    train_batch = cfg['train_batch']
    test_batch = cfg['test_batch']


    #paths for data
    base_path = cfg['base_path']
    data_path = cfg['data_path']
    result_path = cfg['result_path']
    pre_trained_path = cfg['pre_trained_path']
    model_name = cfg['model_name']
    preprocessed = cfg['preprocessed']

    result_file = result_path + 'check_point_' + model_name + '.ckpt'

    #rnn params
    clip_norm = cfg['clip_norm']

    #quantization params
    n_bits_dict = cfg['n_bits_dict']

    #internal args for early stopping
    cur_lr = initial_lr
    cur_patience = 0
    cur_decay_time = 0
    best_valid_cer = 100

    #save configure
    with open(result_path + 'configure.pkl', 'wb') as f:
        pickle.dump(cfg, f)

    #load dataset
    dataset = WSJDataSet(train_batch, charset, base_path, data_path = data_path, preprocessed = preprocessed)



    with tf.Graph().as_default():
        # graph construct
        model = ctc_model_fixed(
                model_name, kernel_size, lstm_dim, lstm_n_layer, 
                opt_name, am_output_dim, dr_ratio,
                n_bits_dict, pre_trained_path,
                )

        #input for the graph
        with tf.variable_scope('PLACEHOLDER'):
            x = tf.placeholder(tf.float32, [None, None, 40, 3], 'x')    # [batch_size, max_seq_len, 40, 3]
            y = tf.sparse_placeholder(tf.int32, name = 'y')                    #sparse placeholder
            seq_len = tf.placeholder(tf.int32, [None], name = 'seq_len')        # [batch_size]
            phase = tf.placeholder(tf.bool, [], name = 'phase')            #scalar
            new_lr = tf.placeholder(tf.float32, [], name = 'new_lr')

        #connect loss calc graph
        logit= model.get_logit(x, seq_len, phase)
        loss, cer = model.get_ctc_loss(logit, seq_len, y)

        #initialize fixed_model
        model.init_quantize()

        print('step size')
        for k in model.step_size.keys():
            print(k, ' : ', model.step_size[k])

        print('n_bits')
        for k in model.n_bits.keys():
            print(k, ' : ', model.n_bits[k])


        with tf.variable_scope('lr'):
            lr = tf.Variable(cur_lr, trainable=False)
        lr_update_op = tf.assign(lr, new_lr)

        #define optimizer
        if opt_name == "Adam":
            with tf.variable_scope(opt_name):
                opt = tf.train.AdamOptimizer(lr)
        else:
            print("change optimizer to defined cfg['opt_name'] : ", opt_name)
            raise NotImplementedError
        
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients, _ = zip(*opt.compute_gradients(loss, model.params))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
            train_op = opt.apply_gradients(zip(gradients, model.grad_storage))

        quantize_op = model.quantize_op()

        #define saver
        saver = tf.train.Saver()

        #history
        train_loss_hist = []
        train_cer_hist = []
        valid_loss_hist = []
        valid_cer_hist = []
        valid_wer_hist = []

        #begin training
        epoch = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            #load pre-trained params
            model.load_all_params(sess)

            #initial quantize(direct_quantize)
            _ = sess.run(quantize_op)

            status = 'save_param'

            #curriculum learning control
            while epoch < max_epoch:
                print('--------', epoch, '/', max_epoch, '--------')
                start_time = time.time()
                dataset.set_batch_size(train_batch)
                dataset.set_mode('train_all')

                #early stopping control
                if status == 'end_train':
                    time.sleep(1)
                    saver.restore(sess, result_file)
                    break
                elif status == 'change_lr':
                    time.sleep(1)
                    saver.restore(sess, result_file)
                    cur_lr *= decay_factor
                    cur_patience = 0
                    cur_decay_time +=1
                    sess.run(lr_update_op, feed_dict = {new_lr : cur_lr})
                    print('lr changed to : ', cur_lr)
                    print('current decay : ', cur_decay_time, " / ", decay_time)
                elif status == 'save_param':
                    cur_patience = 0
                    saver.save(sess, result_file)
                elif status == 'keep_train':
                    cur_patience +=1
                elif status == 'roll_back':
                    saver.restore(sess, result_file)
                    cur_patience = 0
                    status = 'keep_train'
                else:
                    raise NotImplementedError


                #train
                epoch_loss = []
                epoch_cer = []
                with tf.name_scope('train'):
                    train_phase = True

                    while dataset.iter_flag():
                        batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape, _ = dataset.get_data()
                        cur_loss, cur_cer, _ = sess.run(
                            [loss, cer, train_op],
                            feed_dict = {x : batch_x,
                                         seq_len : batch_seq_len,
                                         y : (sparse_indices, sparse_values, sparse_shape),
                                         phase : train_phase}
                        )
                        _ = sess.run(quantize_op)
                        if np.isnan(cur_loss):
                            status = 'roll_back'
                            epoch-=1
                            print('nan train err detected. epoch will be roll-backed')
                            break
                        elif cur_loss != np.inf:
                            epoch_loss.append(cur_loss)
                        else:
                            #print('infinite loss for batch',dataset.counter-1)
                            #print('file name : ', dataset.train_list[dataset.counter-1][:-1])
                            pass
                        epoch_cer.append(cur_cer)

                        if (dataset.counter % 1000) == 0:
                            print('cur batch : ', dataset.counter,'/',dataset.n_batch, 'loss : ', cur_loss, ', cer : ', cur_cer)

                    if status == 'roll_back':
                        continue

                    epoch_loss = np.mean(np.asarray(epoch_loss, dtype='float32'))
                    epoch_cer = np.mean(np.asarray(epoch_cer, dtype='float32'))
                    train_loss_hist.append(epoch_loss)
                    train_cer_hist.append(epoch_cer)

                # evaluation
                epoch_loss = []
                epoch_cer = []
                epoch_wer = []
                dataset.set_batch_size(test_batch)
                dataset.set_mode('valid')

                with tf.name_scope('valid'):
                    train_phase = False
                    while dataset.iter_flag():
                        batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape, label_string = dataset.get_data()
                        cur_loss, cur_cer, cur_logit = sess.run(
                            [loss, cer, logit],
                            feed_dict={x: batch_x,
                                       seq_len: batch_seq_len,
                                       y: (sparse_indices, sparse_values, sparse_shape),
                                       phase: train_phase}
                        )
                        label = label_string[0][:-1] + ' <\s>'
                        predict = greedy_decoding(np.argmax(cur_logit, axis = -1), charset)
                        
                        epoch_wer.append(WER(label, predict))                        
                        epoch_loss.append(cur_loss)
                        epoch_cer.append(cur_cer)
                    epoch_loss = np.mean(np.asarray(epoch_loss, dtype='float32'))
                    epoch_cer = np.mean(np.asarray(epoch_cer, dtype='float32'))
                    epoch_wer = np.mean(np.asarray(epoch_wer, dtype='float32'))
                    valid_loss_hist.append(epoch_loss)
                    valid_cer_hist.append(epoch_cer)
                    valid_wer_hist.append(epoch_wer)

                #early stopping
                if epoch_cer > best_valid_cer:
                    if cur_patience == patience:
                        if cur_decay_time == decay_time:
                            status = 'end_train'
                        else:
                            status = 'change_lr'
                    else:
                        status = 'keep_train'
                else:
                    status = 'save_param'
                    best_valid_cer = epoch_cer

                end_time = time.time()


                print('train loss - ', train_loss_hist[-1], ' | cer - ', train_cer_hist[-1])
                print('valid loss - ', valid_loss_hist[-1], ' | cer - ', valid_cer_hist[-1])
                print('valid wer - ', valid_wer_hist[-1])
                print('status : ', status, ', training time : ', end_time - start_time)
                print('label    : |',label,'|')
                print('predict  : |',predict,'|')
                epoch+=1


            #final test
            # evaluation
            epoch_loss = []
            epoch_cer = []
            epoch_wer = []
            dataset.set_batch_size(test_batch)
            dataset.set_mode('test')
            start_time = time.time()
            with tf.name_scope('test'):
                train_phase = False
                while dataset.iter_flag():
                    batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape, label_string = dataset.get_data()
                    cur_loss, cur_cer, cur_logit = sess.run(
                        [loss, cer, logit],
                        feed_dict={x: batch_x,
                                   seq_len: batch_seq_len,
                                   y: (sparse_indices, sparse_values, sparse_shape),
                                   phase: train_phase}
                    )
                    label = label_string[0][:-1] + ' <\s>'
                    predict = greedy_decoding(np.argmax(cur_logit, axis = -1), charset)
                        
                    epoch_wer.append(WER(label, predict))                        
                    epoch_loss.append(cur_loss)
                    epoch_cer.append(cur_cer)
                epoch_loss = np.mean(np.asarray(epoch_loss, dtype='float32'))
                epoch_cer = np.mean(np.asarray(epoch_cer, dtype='float32'))
                epoch_wer = np.mean(np.asarray(epoch_wer, dtype='float32'))
            end_time = time.time()
            print('\n\n--------final result for eval92 test data--------')
            print('loss - ', epoch_loss, ' | cer - ', epoch_cer, ' | wer - ',epoch_wer)
            print('test set inference time : ', end_time - start_time)
            model.save_all_params(sess, result_path)


        #save results
        np.save(result_path + model_name + '_train_loss.npy', train_loss_hist)
        np.save(result_path + model_name + '_train_cer.npy', train_cer_hist)
        np.save(result_path + model_name + '_valid_loss.npy', valid_loss_hist)
        np.save(result_path + model_name + '_valid_cer.npy', valid_cer_hist)
        np.save(result_path + model_name + '_valid_wer.npy', valid_wer_hist)


if __name__ == '__main__':
    cfg = config_fixed()
    train(cfg)
