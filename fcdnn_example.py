import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import time
from matplotlib import pyplot as plt


from dataset import MNISTData


def main():

    learning_rate = 0.001
    batch_size = 100
    n_epoch = 30
    dataset = MNISTData(batch_size = batch_size, data_shape='flatten')
    save_path = 'results/mlp/'


    ####model define
    #input
    with tf.variable_scope("PLACEHOLDER"):
        x = tf.placeholder(tf.float32, [None, 784], 'x')
        y = tf.placeholder(tf.int32, [None, ], 'y')

    lr = tf.Variable(learning_rate, trainable=False)

    #784-512-512-10
    h = tf.layers.dense(x, 512)
    h = tf.nn.relu(h)
    h = tf.layers.dense(h, 512)
    h = tf.nn.relu(h)
    logit = tf.layers.dense(h, 10)

    #set cost function
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logit)
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(tf.argmax(logit, axis=-1)), y)))

    #define optimizer
    opt = tf.train.GradientDescentOptimizer(lr)

    #training operation
    train_op = opt.minimize(loss)



    train_loss_hist = []
    train_acc_hist = []
    valid_loss_hist = []
    valid_acc_hist = []
    # training
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())  #initialize variables
        '''
        for pp in tf.trainable_variables():
            print(pp.name)
        exit(0)
        '''

        for epoch in range(n_epoch):
            start_time = time.time()
            dataset.set_mode('train')

            # train procedure
            epoch_loss = []
            epoch_acc = []

            while dataset.iter_flag():
                batch_x, batch_y = dataset.get_data()
                cur_loss, cur_acc, _ = session.run([loss, acc, train_op],
                                           feed_dict = {x : batch_x, y : batch_y}
                                           )
                epoch_loss.append(cur_loss)
                epoch_acc.append(cur_acc)

            epoch_loss = np.mean(np.asarray(epoch_loss, dtype='float32'))
            epoch_acc = np.mean(np.asarray(epoch_acc, dtype='float32'))

            train_loss_hist.append(epoch_loss)
            train_acc_hist.append(epoch_acc)

            #evaluation procedure
            dataset.set_mode('valid')
            epoch_loss = []
            epoch_acc = []

            while dataset.iter_flag():
                batch_x, batch_y = dataset.get_data()
                cur_loss, cur_acc = session.run([loss, acc],
                                           feed_dict = {x : batch_x, y : batch_y}
                                           )
                epoch_loss.append(cur_loss)
                epoch_acc.append(cur_acc)

            epoch_loss = np.mean(np.asarray(epoch_loss, dtype='float32'))
            epoch_acc = np.mean(np.asarray(epoch_acc, dtype='float32'))

            valid_loss_hist.append(epoch_loss)
            valid_acc_hist.append(epoch_acc)
            end_time = time.time()

            print('--------', epoch, '/',n_epoch,'--------')
            print('train loss - ', train_loss_hist[-1], ' | acc - ', train_acc_hist[-1])
            print('valid loss - ', valid_loss_hist[-1], ' | acc - ', valid_acc_hist[-1])
            print('training time : ', end_time - start_time)


        #final result with test set
        print('---------training done--------')
        epoch_loss = []
        epoch_acc = []
        dataset.set_mode('test')
        while dataset.iter_flag():
            batch_x, batch_y = dataset.get_data()
            cur_loss, cur_acc = session.run([loss, acc],
                                       feed_dict={x: batch_x, y: batch_y}
                                       )
            epoch_loss.append(cur_loss)
            epoch_acc.append(cur_acc)

        epoch_loss = np.mean(np.asarray(epoch_loss, dtype='float32'))
        epoch_acc = np.mean(np.asarray(epoch_acc, dtype='float32'))

        print('final result with test data')
        print('loss : ', epoch_loss, 'acc : ', epoch_acc)

        #save model
        for pp in tf.all_variables():
            n_arr = pp.name.split('/')
            param_name = "_".join(n_arr)
            v = session.run(pp)
            np.save(save_path+param_name+'.npy', v)
        print('results is saved in ', save_path)



    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(train_loss_hist)
    axarr[0].set_title('training loss')
    axarr[1].plot(valid_loss_hist)
    axarr[1].set_title('validation loss')

    plt.show()


if __name__ == '__main__':
    main()


