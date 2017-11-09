
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