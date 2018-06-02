class FLAGS(object):

    """
    General settings
    """
    input_size = 512
    heatmap_size = 64
    cpm_stages = 6
    num_of_joints = 24
    color_channel = 'RGB'
    joint_gaussian_variance = 1.0
    normalize_img = True
    use_gpu = True
    gpu_id = '0'
    if_show = True
    img_show_iters = 100

    """
       Training settings
    """
    network_def = 'cpm_body'

    pretrained_model = './data_gu_stage_6'
    datagenerator_config_file = './preprocess/config.cfg'

    batch_size = 8

    init_lr = 0
    lr_decay_rate = 0
    lr_decay_step = 0

    total_epoch = 40
    total_num = 40000
    training_iters = int(total_num * total_epoch / batch_size)
    validation_iters = 500
    model_save_iters = 10000