import tensorflow as tf
import numpy as np

def get_step_size(w, n_bits, lstm_flag = False, lstm_dim = -1):
    s = 0.00001
    th  = 1e-6
    hist = [s]
    M = 2**(n_bits) - 1

    if lstm_flag:
        w1, w2 = np.split(w, [-lstm_dim])
        s1 = s
        s2 = s
        for i in range(100):
            z = np.sign(w1) * np.minimum(np.round(np.abs(w1) / s1), (M - 1) / 2)
            s1 = np.sum(w1 * z) / np.sum(z ** 2)
            hist.append(s1)
            if np.abs(hist[i] - hist[i + 1]) < th:
                break
        hist = [s2]
        for i in range(100):
            z = np.sign(w2) * np.minimum(np.round(np.abs(w2) / s2), (M - 1) / 2)
            s2 = np.sum(w2 * z) / np.sum(z ** 2)
            hist.append(s2)
            if np.abs(hist[i] - hist[i + 1]) < th:
                break

        return [s1, s2]

    else:
        for i in range(100):
            z = np.sign(w) * np.minimum(np.round(np.abs(w) / s), (M-1)/2)
            s = np.sum(w*z) / np.sum(z**2)
            hist.append(s)
            if np.abs(hist[i] - hist[i+1]) < th:
                break

        return s




class quantization_model(object):
    def __init__(self, n_bits_dict, pretrained_path, opt_name):
        self.params = []
        self.grad_storage = []
        self.step_size = dict()
        self.n_bits = dict()
        self.step_size_tf = dict()
        self.n_bits_tf = dict()
        self.n_bits_dict = n_bits_dict
        self.saved_path = pretrained_path
        self.lstm_dim = 512
        self.opt_name = opt_name

    def init_quantize(self):
        #create new variables for gradient storage
        for pp in tf.trainable_variables():
            self.params.append(pp)

        for pp in self.params:
            n_arr = pp.name.split(':')
            p_name = n_arr[0] + "_float"
            p = tf.get_variable(p_name, pp.shape, trainable=False)
            self.grad_storage.append(p)
            print(p_name, 'added')

        #assign n_bits and step_size for quantization
        for pp in self.params:
            n_flag = False
            for k in self.n_bits_dict.keys():
                if pp.name.find(k) != -1:
                    if pp.name.find(self.opt_name) != -1:
                        pass
                    else:
                        self.n_bits[pp.name] = self.n_bits_dict[k]
                        n_flag = True
            if not n_flag:
                self.n_bits[pp.name] = -1
            else:
                pass

            if self.n_bits[pp.name] == -1:
                self.step_size[pp.name] = -1.0
            else:
                n_arr = pp.name.split('/')
                n = "_".join(n_arr) + '.npy'
                v = np.load(self.saved_path + n)
                if pp.name.find('lstm') ==-1:
                    lstm_flag = False
                else:
                    lstm_flag = True
                self.step_size[pp.name] = get_step_size(v, self.n_bits[pp.name], lstm_flag, self.lstm_dim)



    def load_all_params(self, sess, path = None):
        #load pre-trained parameters and assign them to corresponding gradients storage
        if path == None:
            saved_path = self.saved_path
        else:
            saved_path = path

        for pp in tf.global_variables():
            if pp.name.find('_float') != -1:
                p_name = pp.name.replace("_float", "")
            else:
                p_name = pp.name

            if p_name.find('lr') != -1 or p_name.find('Variable') != -1:
                continue

            n_arr = p_name.split('/')
            n = "_".join(n_arr) + '.npy'
            v = np.load(saved_path+n)
            sess.run(tf.assign(pp, v))
            print(pp.name, ' loaded')

        #for pp, gg in zip(self.params, self.grad_storage):
        #    v = sess.run(pp)
        #    sess.run(tf.assign(gg, v))

    def save_all_params(self, sess, path):
        for pp in tf.global_variables():
            n_arr = pp.name.split('/')
            n = "_".join(n_arr) + '.npy'
            v = sess.run(pp)
            np.save(path+n, v)
            print(pp.name, ' is saved at ', path+n)




    def quantize_op(self):
        ops = []
        for pp, gg in zip(self.params, self.grad_storage):
            s = self.step_size[pp.name]
            n = self.n_bits[pp.name]

            precision = tf.to_float(tf.pow(2, n - 1) - 1)

            if pp.name.find('lstm') == -1:
                if n == -1:
                    v = gg
                else:
                    v = tf.multiply(tf.minimum(tf.maximum(tf.round(tf.divide(gg, s)), -precision), precision), s)

            else:
                if n==-1:
                    v = gg
                else:
                    s_x = s[0]
                    s_h = s[1]
                    gg1, gg2 = tf.split(gg, [-1, self.lstm_dim], axis=0)

                    v1 = tf.multiply(tf.minimum(tf.maximum(tf.round(tf.divide(gg1, s_x)), -precision), precision), s_x)
                    v2 = tf.multiply(tf.minimum(tf.maximum(tf.round(tf.divide(gg2, s_h)), -precision), precision), s_h)

                    v = tf.concat([v1, v2], axis=0)

            ops.append(tf.assign(pp, v))
        return ops