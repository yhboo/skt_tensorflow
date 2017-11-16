import numpy as np
import scipy.io as sio



class MNISTData(object):
    def __init__(self, batch_size=100, valid_ratio = 0.1, data_path = './data/', data_shape = 'flatten', seed = 1234):

        #save parameters
        self.batch_size = batch_size
        self.valid_ratio = valid_ratio
        self.data_shape = data_shape
        np.random.seed(seed)

        #define inner args for training
        self.mode = 'train'         # string, ('train', 'valid', 'test')
        self.counter = 0
        self.n_batch = 0
        self.n_data = 100
        self.random_idx_arr = np.random.permutation(self.n_data)

        #load .mat file
        matdata = sio.loadmat(data_path+'mnist_all.mat')
        #for i in range(10):
        #    key = 'test'+str(i)
        #    print(i,'th -', matdata[key].shape)

        #change matdata to numpy type data
        self.train_x = matdata['train0']
        self.train_y = np.zeros([self.train_x.shape[0], ], dtype = 'int32')
        self.test_x = matdata['test0']
        self.test_y = np.zeros([self.test_x.shape[0], ], dtype = 'int32')
        for i in range(1, 10):
            train_key = 'train'+str(i)
            test_key = 'test'+str(i)
            self.train_x = np.concatenate((self.train_x, matdata[train_key]), axis = 0)
            self.train_y = np.concatenate((self.train_y, i*np.ones([matdata[train_key].shape[0], ], dtype = 'int32')), axis = 0)
            self.test_x = np.concatenate((self.test_x, matdata[test_key]), axis=0)
            self.test_y = np.concatenate((self.test_y, i*np.ones([matdata[test_key].shape[0], ], dtype='int32')), axis = 0)

        self.train_x = (self.train_x - 127.0) / 255.0
        self.test_x = (self.test_x - 127.0) / 255.0


        #reshape data for corresponding model
        if self.data_shape == 'flatten':
            pass
        elif self.data_shape == 'image':
            self.train_x = np.reshape(self.train_x, [-1, 28, 28, 1])
            self.test_x = np.reshape(self.test_x, [-1, 28, 28, 1])
        else:
            print("data shape must be 'flatten' or 'image'")
            raise NotImplementedError


        #split validation data
        n_train = self.train_x.shape[0]
        if self.valid_ratio > 0:
            n_valid = np.floor(n_train * self.valid_ratio).astype('int32')
            rand_idx = np.random.permutation(n_train)
            self.valid_x = self.train_x[rand_idx[0 : n_valid]]
            self.valid_y = self.train_y[rand_idx[0 : n_valid]]
            self.train_x = self.train_x[rand_idx[n_valid : ]]
            self.train_y = self.train_y[rand_idx[n_valid : ]]

        self.n_data = self.train_x.shape[0]


    #set current mode
    def set_mode(self, mode):
        self.mode = mode

        if self.mode == 'train':
            self.n_data = self.train_x.shape[0]
        elif self.mode == 'valid':
            self.n_data = self.valid_x.shape[0]
        elif self.mode == 'test':
            self.n_data = self.test_x.shape[0]

        self.reset()

    #reset data iteration
    def reset(self):
        self.counter = 0
        self.random_idx_arr = np.random.permutation(self.n_data)
        self.n_batch = int(self.n_data / self.batch_size)

    #change batch size
    def set_batch_size(self, new_batch):
        self.batch_size = new_batch
        self.reset()

    #iteration flag for each epoch
    def iter_flag(self):
        if self.counter < self.n_batch:
            return True
        else:
            return False


    def get_data(self):
        cur_idx_begin = self.counter*self.batch_size
        data_idx = self.random_idx_arr[cur_idx_begin : cur_idx_begin + self.batch_size]

        if self.mode == 'train':
            batch_x = self.train_x[data_idx]
            batch_y = self.train_y[data_idx]
        elif self.mode == 'valid':
            batch_x = self.valid_x[data_idx]
            batch_y = self.valid_y[data_idx]
        elif self.mode == 'test':
            batch_x = self.test_x[data_idx]
            batch_y = self.test_y[data_idx]
        else:
            raise NotImplementedError

        self.counter += 1
        return batch_x, batch_y


    #dataset summary
    def summary(self):
        print('---------- dataset summary ----------')
        print('data type        : ', self.data_shape)
        print('batch size       : ', self.batch_size)
        print('valid ratio      : ', self.valid_ratio)
        print('train_x shape : ', self.train_x.shape)
        print('train_y shape : ', self.train_y.shape)
        if self.valid_ratio > 0:
            print('valid_x shape : ', self.valid_x.shape)
            print('valid_y shape : ', self.valid_y.shape)
        print('test_x shape  : ,', self.test_x.shape)
        print('test_y shape  : ,', self.test_y.shape)













if __name__ == '__main__':
    batch_size = 100
    valid_ratio = 0.1
    data_path = './data/'

    dataset = MNISTData(batch_size, valid_ratio, data_path)
    dataset.set_mode('train')
    x, y = dataset.get_data()

    print(x.shape)
    print(np.max(x))
    print(np.min(x))
    print(type(x))
    print(y.shape)
    print(np.max(y))
    print(np.min(y))
    print(type(y))

    '''
    dataset.summary()
    dataset.set_mode('test')
    counter_iter = 0
    counter_label = np.zeros([10,], dtype='int32')
    while(dataset.iter_flag()):
        x, y = dataset.get_data()
        #for i in range(y.shape[0]):
        #    counter_label[y[i]]+=1
        counter_label[y[0]] +=1
        counter_iter+=1
    print(x.shape)

    print('counter_iter : ', counter_iter)
    print('label counter')
    print(counter_label)
    '''
