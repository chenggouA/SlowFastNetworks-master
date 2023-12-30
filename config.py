params = dict()

params['num_classes'] = 2

# params['dataset'] = './dataset'
params['dataset'] = 'F:/data/fight/HockeyFightVidoes/data'


params['epoch_num'] = 40
params['batch_size'] = 6
params['step'] = 10
params['num_workers'] = 1
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['display'] = 10
params['pretrained'] = None
params['gpu'] = [0]
params['log'] = 'log'
params['save_path'] = 'fightDetection'
params['clip_len'] = 64 # 处理的帧数
params['frame_sample_rate'] = 1
