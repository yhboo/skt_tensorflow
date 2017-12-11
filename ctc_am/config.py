
def config_cnn():
    cfg = dict()
    cfg['mode'] = 'uni'
    cfg['gpu_target'] = "3"
    cfg['charset'] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
    cfg['initial_lr'] = 3e-4
    cfg['decay_factor'] = 0.2
    cfg['decay_time'] = 6
    cfg['patience'] = 8
    cfg['max_epoch'] = 1000
    cfg['train_batch'] = 30
    #cfg['max_epoch'] = 1
    #cfg['train_batch'] = 1
    cfg['test_batch'] = 1
    cfg['base_path'] = '/home/yhboo/skt/wavectrl/raw_wav/'
    cfg['data_path'] = './data/'
    cfg['result_path'] = './results/ctc_cnn/train_results/'
    cfg['model_name'] = 'ctc_cnn_online'
    cfg['clip_norm'] = 400

    cfg['epoch_small'] = 10
    cfg['epoch_mid'] = 30
    cfg['epoch_big'] = 60
    #cfg['epoch_small'] = 0
    #cfg['epoch_mid'] = 0
    #cfg['epoch_big'] = 0
    cfg['kernel_size'] = (10,3)
    cfg['lstm_dim'] = 512
    cfg['n_layer'] = 3
    cfg['opt_name'] = 'Adam'
    cfg['dropout_ratio'] = 0.5
    cfg['preprocessed'] = True

    print('training configuration summary')
    for k in cfg.keys():
        print(k, ':', cfg[k])

    return cfg

def config_bi():
    cfg = dict()
    cfg['mode'] = 'bi'
    cfg['gpu_target'] = "2"
    cfg['charset'] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
    cfg['initial_lr'] = 3e-4
    cfg['decay_factor'] = 0.2
    cfg['decay_time'] = 6
    cfg['patience'] = 8
    cfg['max_epoch'] = 1000
    cfg['train_batch'] = 30
    #cfg['max_epoch'] = 1
    #cfg['train_batch'] = 1
    cfg['test_batch'] = 1
    cfg['base_path'] = '/home/yhboo/skt/wavectrl/raw_wav/'
    cfg['data_path'] = './data/'
    cfg['result_path'] = './results/ctc_bi2/train_results/'
    cfg['model_name'] = 'ctc_bi'
    cfg['clip_norm'] = 400

    cfg['epoch_small'] = 10
    cfg['epoch_mid'] = 30
    cfg['epoch_big'] = 60
    #cfg['epoch_small'] = 0
    #cfg['epoch_mid'] = 0
    #cfg['epoch_big'] = 0
    cfg['kernel_size'] = 3
    cfg['lstm_dim'] = 512
    cfg['n_layer'] = 3
    cfg['opt_name'] = 'Adam'
    cfg['dropout_ratio'] = 0.5
    cfg['preprocessed'] = True

    print('training configuration summary')
    for k in cfg.keys():
        print(k, ':', cfg[k])

    return cfg

def config_fixed():
    cfg = dict()
    cfg['name'] = 'ctc_fixed_configuration'
    cfg['charset'] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
    cfg['initial_lr'] = 5e-7
    cfg['decay_factor'] = 0.2
    cfg['decay_time'] = 3
    cfg['patience'] = 5
    cfg['max_epoch'] = 200
    cfg['train_batch'] = 40
    #cfg['max_epoch'] = 1
    # cfg['train_batch'] = 1
    cfg['test_batch'] = 1
    cfg['base_path'] = '/home/yhboo/skt/wavectrl/raw_wav/'
    cfg['data_path'] = './data/'
    cfg['pre_trained_path'] = './results/ctc_model2/train_results/'
    cfg['result_path'] = './results/ctc_model2/retrain_results/'
    cfg['model_name'] = 'ctc'
    cfg['clip_norm'] = 200

    n_bits_dict = dict()
    n_bits_dict['lstm_cell/weights'] = 8
    n_bits_dict['dense/kernel']= 8
    cfg['n_bits_dict'] = n_bits_dict
    cfg['kernel_size'] = 10
    cfg['lstm_dim'] = 512
    cfg['n_layer'] = 3
    cfg['opt_name'] = 'Adam'
    cfg['dropout_ratio'] = 0.5
    cfg['preprocessed'] = True


    print('training configuration summary')
    for k in cfg.keys():
        print(k, ':', cfg[k])

    return cfg

def config_example():
    cfg = dict()
    cfg['mode'] = 'uni'
    cfg['gpu_target'] = "0"
    cfg['charset'] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
    cfg['initial_lr'] = 3e-4
    cfg['decay_factor'] = 0.2
    cfg['decay_time'] = 6
    cfg['patience'] = 8
    cfg['max_epoch'] = 10
    cfg['train_batch'] = 40
    #cfg['max_epoch'] = 1
    #cfg['train_batch'] = 1
    cfg['test_batch'] = 1
    #cfg['base_path'] = './data/data_example/'
    cfg['base_path'] = './data/data_example_processed/'
    cfg['data_path'] = './data/'
    cfg['result_path'] = './results/ctc_example/train_results/'
    cfg['model_name'] = 'ctc_cnn_online'
    cfg['clip_norm'] = 400

    cfg['epoch_small'] = 0
    cfg['epoch_mid'] = 0
    cfg['epoch_big'] = 0
    #cfg['epoch_small'] = 0
    #cfg['epoch_mid'] = 0
    #cfg['epoch_big'] = 0
    cfg['kernel_size'] = (10,3)
    cfg['lstm_dim'] = 512
    cfg['n_layer'] = 3
    cfg['opt_name'] = 'Adam'
    cfg['dropout_ratio'] = 0.5
    cfg['preprocessed'] = True

    print('training configuration summary')
    for k in cfg.keys():
        print(k, ':', cfg[k])

    return cfg
