#!/usr/bin/python2
import utils
import config as cfg
from CNN_models import models
import os
import numpy as np
from img_process import load_data as ld
import sys

if __name__ == '__main__':
    np.random.seed(2345)
    #--model files to run --
    #net_version, tag, fold, epoch
    #m_param = [(1,1,0,5),(1,1,4,5)];

    #m_param = [(cfg.net_version,cfg.tag,cfg.folds[-1],cfg.num_epochs)];
    #fname = 'tag{}_ep{}'.format(cfg.tag, cfg.num_epochs)
    epoch = int(sys.argv[1])
    m_param = [(cfg.net_version,cfg.tag,cfg.folds[-1], epoch)];
    fname = 'tag{}_ep{}'.format(cfg.tag, epoch)

    model_weights = [1.0]*len(m_param);

    file_fmt = cfg.params_dir + '/cnn{}_tag{}_f{}_ep{}.npz';
    shape = (None, 3, cfg.WIDTH, cfg.HEIGHT)
    predict_fn = models.get_predict_function(m_param, model_weights, file_fmt, shape);

    load_and_process = ld.LoadAndProcess(
            size = (cfg.WIDTH, cfg.HEIGHT),
            augmentation_params = None,
            crop = None,
            color_noise = 0,
            fill_size = cfg.pretrained);

    batch_size = cfg.batch_size;
    test_imgs,test_labels = ld.list_imgs_labels(cfg.data_dir,data='test');
    test_data = ld.ImgStream(test_imgs, test_labels, batch_size,
            cycle=False, file_dir_fmt=cfg.data_dir+'/test/{}',
            load_and_process = load_and_process, preload=None);

    print("num of test cases: {}".format(len(test_data)));

    res = [];
    c = 0;
    for imgs,labels in test_data:
        res.append(predict_fn(imgs));
        c += 1;
        if c%50 == 0:
            print("{} processed ".format(c*batch_size));

    res = np.concatenate(res);
    filename = cfg.output_dir + "/submit_{}.csv".format(fname);
    print(res[-1])
    utils.make_submission(filename, test_imgs, res, 0.5e-3);
