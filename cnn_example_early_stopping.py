import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from dataset import MNISTData




def main():
    learning_rate = 0.0002
    batch_size = 100
    n_epoch = 100
    max_change = 3
    max_patience = 3
    dataset = MNISTData(batch_size=batch_size, data_shape='image')
    save_param_path = 'results/LeNet-5_ES/'
    save_path = save_param_path + 'best_model.ckpt'


    ####model define
    # input
    with tf.variable_scope("PLACEHOLDER"):
        x = tf.placeholder(tf.float32, [None, 28, 28, 1], 'x')
        y = tf.placeholder(tf.int32, [None, ], 'y')
        new_lr = tf.placeholder(tf.float32, [], 'New_lr')

    lr = tf.Variable(learning_rate, trainable=False, name='learning_rate')

    # LeNet-5 (with BN)
    C1 = tf.layers.conv2d(x, filters=6, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)
    C1 = tf.nn.relu(C1)
    S2 = tf.layers.max_pooling2d(C1, pool_size=(2, 2), strides=(2, 2))
    C3 = tf.layers.conv2d(S2, filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', use_bias=False)
    C3 = tf.nn.relu(C3)
    S4 = tf.layers.max_pooling2d(C3, pool_size=(2, 2), strides=(2, 2))
    S4 = tf.reshape(S4, [-1, 16 * 5 * 5])

    C5 = tf.layers.dense(S4, 120)
    C5 = tf.nn.relu(C5)
    F6 = tf.layers.dense(C5, 84)
    F6 = tf.nn.relu(F6)
    logit = tf.layers.dense(F6, 10)

    # set cost function
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logit)
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(tf.argmax(logit, axis=-1)), y)))


    lr_update_op = tf.assign(lr, new_lr)


    # define optimizer
    opt = tf.train.GradientDescentOptimizer(lr)
    saver = tf.train.Saver()

    # training operation
    train_op = opt.minimize(loss)



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        best_valid_loss = 1000000
        patience = 0
        change = 0
        status = 'save_param'

        for epoch in range(n_epoch):
            print('... Epoch', epoch, status)
            start_time = time.clock()
            if status == 'end_train':
                time.sleep(5)
                saver.restore(sess, save_path)
                break
            elif status == 'change_lr':
                time.sleep(5)
                saver.restore(sess, save_path)
                sess.run(lr_update_op, feed_dict={new_lr: 0.01 * np.power(0.1, change)})
            elif status == 'save_param':
                saver.save(sess, save_path)
            else:
                pass

            epoch_loss = []
            epoch_acc = []
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
            print('...... Train loss', epoch_loss)
            print('...... Train accuracy', epoch_acc)

            epoch_loss = []
            epoch_acc = []
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
            print('...... Valid loss', epoch_loss)
            print('...... Valid best loss', best_valid_loss)
            print('...... Valid accuracy', epoch_acc)

            if epoch_loss > best_valid_loss:
                patience += 1
                print('......... Current patience', patience)
                if patience >= max_patience:
                    change += 1
                    patience = 0
                    print('......... Current lr change', change)
                    if change >= max_change:
                        status = 'end_train'  # (load param, stop training)
                    else:
                        status = 'change_lr'  # (load param, change learning rate)
                else:
                    status = 'keep_train'  # (keep training)
            else:
                best_valid_loss = epoch_loss
                patience = 0
                print('......... Current patience', patience)
                status = 'save_param'  # (save param, keep training)

            end_time = time.clock()
            print('...... Time:', end_time - start_time)





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

        print('final result with test data')
        print('loss : ', epoch_loss, 'acc : ', epoch_acc)




if __name__ == '__main__':
    main()

