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



    ####model define
    #input


    #784-512-512-10


    #set cost function


    #define optimizer


    #training operation


    train_loss_hist = []
    train_acc_hist = []
    valid_loss_hist = []
    valid_acc_hist = []
    # training
    with tf.Session() as session:


        for epoch in range(n_epoch):
            pass
            start_time = time.time()
            # train procedure
            dataset.set_mode('train')
            epoch_loss = []
            epoch_acc = []

            while dataset.iter_flag():
                pass



            #evaluation procedure
            dataset.set_mode('valid')
            epoch_loss = []
            epoch_acc = []

            while dataset.iter_flag():
                pass

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
            pass



        print('final result with test data')
        print('loss : ', epoch_loss, 'acc : ', epoch_acc)





    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(train_loss_hist)
    axarr[0].set_title('training loss')
    axarr[1].plot(valid_loss_hist)
    axarr[1].set_title('validation loss')

    plt.show()


if __name__ == '__main__':
    main()