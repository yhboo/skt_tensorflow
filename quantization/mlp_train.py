import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import time

from models import mlp
from dataset import MNISTData




def simple_train_with_early_stopping(cfg):
    initial_lr = cfg['learning_rate']
    batch_size = cfg['batch_size']
    save_path = cfg['save_path']
    hist_path = cfg['hist_path']
    max_epoch = cfg['max_epoch']
    decay_time = cfg['max_change']
    max_patience = cfg['max_patience']
    decay_factor = cfg['decay_factor']


    dataset = MNISTData(batch_size= batch_size, data_path = '../data/', data_shape = 'flatten')

    cur_lr = initial_lr
    cur_patience = 0
    cur_decay_time = 0
    best_valid_acc = 0



    with tf.Graph().as_default():
        ###graph construct
        # input
        with tf.variable_scope("PLACEHOLDER"):
            x = tf.placeholder(tf.float32, [None, 784], 'x')
            y = tf.placeholder(tf.int32, [None, ], 'y')
            new_lr = tf.placeholder(tf.float32, [], 'new_lr')

        #model
        model = mlp('mlp')
        logit = model.get_logit(x)

        #cost function
        loss, acc = model.compute_loss_acc(logit, y)

        #define optimizer
        lr = tf.Variable(initial_lr, trainable=False, name = 'lr')
        opt = tf.train.AdamOptimizer(lr)

        #define update ops
        lr_update_op = tf.assign(lr, new_lr)

        train_op = opt.minimize(loss)

        #define saver
        saver = tf.train.Saver()

        ###graph contruct end


        train_loss_hist = []
        train_acc_hist = []
        valid_loss_hist = []
        valid_acc_hist = []

        #training begins
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


            status = 'keep_train'

            for epoch in range(max_epoch):
                print('--------', epoch, '/', max_epoch, '--------')
                start_time = time.time()
                # early stopping control
                if status == 'end_train':
                    time.sleep(1)
                    #model.load_params(sess, save_path)
                    saver.restore(sess, save_path)
                    break
                elif status == 'change_lr':
                    time.sleep(1)
                    #model.load_params(sess, save_path)
                    saver.restore(sess, save_path)
                    cur_lr *= decay_factor
                    cur_patience = 0
                    cur_decay_time += 1
                    sess.run(lr_update_op, feed_dict={new_lr: cur_lr})
                    print('lr changed to : ', cur_lr)
                    print('current decay : ', cur_decay_time, " / ", decay_time)
                elif status == 'save_param':
                    cur_patience = 0
                    #model.save_params(sess, save_path)
                    saver.save(sess, save_path)
                elif status == 'keep_train':
                    cur_patience += 1
                else:
                    raise NotImplementedError

                epoch_loss = []
                epoch_acc = []
                #training
                with tf.name_scope('train'):
                    dataset.set_mode('train')

                    while dataset.iter_flag():
                        bx, by = dataset.get_data()
                        current_loss, current_accuracy, _ = sess.run(
                            [loss, acc, train_op],
                            feed_dict={x: bx, y: by})
                        epoch_loss.append(current_loss)
                        epoch_acc.append(current_accuracy)
                epoch_loss = np.mean(np.asarray(epoch_loss, 'float32'))
                epoch_acc = np.mean(np.asarray(epoch_acc, 'float32'))
                train_loss_hist.append(epoch_loss)
                train_acc_hist.append(epoch_acc)


                epoch_loss = []
                epoch_acc = []
                #evaluation
                with tf.name_scope('valid'):

                    dataset.set_mode('valid')
                    while dataset.iter_flag():
                        bx, by = dataset.get_data()
                        current_loss, current_accuracy = sess.run(
                            [loss, acc],
                            feed_dict={x: bx, y: by})
                        epoch_loss.append(current_loss)
                        epoch_acc.append(current_accuracy)
                epoch_loss = np.mean(np.asarray(epoch_loss, 'float32'))
                epoch_acc = np.mean(np.asarray(epoch_acc, 'float32'))
                valid_loss_hist.append(epoch_loss)
                valid_acc_hist.append(epoch_acc)

                # early stopping
                if epoch_acc < best_valid_acc:
                    if cur_patience == max_patience:
                        if cur_decay_time == decay_time:
                            status = 'end_train'
                        else:
                            status = 'change_lr'
                    else:
                        status = 'keep_train'
                else:
                    status = 'save_param'
                    best_valid_acc = epoch_acc

                end_time = time.time()
                print('train loss - ', train_loss_hist[-1], ' | acc - ', train_acc_hist[-1])
                print('valid loss - ', valid_loss_hist[-1], ' | acc - ', valid_acc_hist[-1])
                print('status : ', status, ', training time : ', end_time - start_time)

            #final evaluation with test set
            epoch_loss = []
            epoch_acc = []
            dataset.set_mode('test')
            while dataset.iter_flag():
                batch_x, batch_y = dataset.get_data()
                cur_loss, cur_acc = sess.run([loss, acc],
                                             feed_dict={x: batch_x, y: batch_y}
                                             )
                epoch_loss.append(cur_loss)
                epoch_acc.append(cur_acc)

            epoch_loss = np.mean(np.asarray(epoch_loss, dtype='float32'))
            epoch_acc = np.mean(np.asarray(epoch_acc, dtype='float32'))

            print('-----------------------------------------')
            print('final result with test data')
            print('loss : ', epoch_loss, 'acc : ', epoch_acc)

            #save results
            np.save(hist_path+'_train_loss.npy', np.asarray(train_loss_hist, dtype = 'float32'))
            np.save(hist_path + '_train_acc.npy', np.asarray(train_acc_hist, dtype='float32'))
            np.save(hist_path + '_valid_loss.npy', np.asarray(valid_loss_hist, dtype='float32'))
            np.save(hist_path + '_valid_acc.npy', np.asarray(valid_acc_hist, dtype='float32'))
            model.save_params(sess, save_path)




if __name__ == '__main__':
    cfg = dict()


    cfg['learning_rate'] = 1e-4
    cfg['batch_size'] = 50
    cfg['save_path'] = './results/mlp_float/params/'
    cfg['hist_path'] = './results/mlp_float/logs/'
    cfg['max_epoch'] = 300
    cfg['max_change'] = 3
    cfg['max_patience'] = 5
    cfg['decay_factor'] = 0.1

    simple_train_with_early_stopping(cfg)