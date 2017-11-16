
def config():
    cfg = dict()
    cfg['charset'] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
    cfg['initial_lr'] = 5e-5
    cfg['decay_factor'] = 0.2
    cfg['decay_time'] = 4
    cfg['patience'] = 10
    cfg['max_epoch'] = 1000
    cfg['train_batch'] = 40
    #cfg['max_epoch'] = 1
    #cfg['train_batch'] = 1
    cfg['test_batch'] = 1
    cfg['base_path'] = '/home/yhboo/skt/wavectrl/raw_wav/'
    cfg['data_path'] = './data/'
    cfg['result_path'] = './results/'
    cfg['model_name'] = 'ctc'
    cfg['clip_norm'] = 200

    cfg['epoch_small'] = 10
    cfg['epoch_mid'] = 30
    cfg['epoch_big'] = 60
    #cfg['epoch_small'] = 0
    #cfg['epoch_mid'] = 0
    #cfg['epoch_big'] = 0

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
    # cfg['max_epoch'] = 1
    # cfg['train_batch'] = 1
    cfg['test_batch'] = 1
    cfg['base_path'] = '/home/yhboo/skt/wavectrl/raw_wav/'
    cfg['data_path'] = './data/'
    cfg['pre_trained_path'] = './results/variables/'
    cfg['result_path'] = './results/fixed_variables/'
    cfg['model_name'] = 'ctc'
    cfg['clip_norm'] = 200

    cfg['epoch_small'] = -1
    cfg['epoch_mid'] = -1
    cfg['epoch_big'] = -1
    n_bits_dict = dict()
    n_bits_dict['lstm_cell_weights'] = 8
    n_bits_dict['dense_kernel']= 8
    cfg['n_bits_dict'] = n_bits_dict


    print('training configuration summary')
    for k in cfg.keys():
        print(k, ':', cfg[k])

    return cfg