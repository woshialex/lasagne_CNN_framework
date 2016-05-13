#!/usr/bin/python2
import utils
import config as cfg
import models
import os
import numpy as np
import load_data as ld

if __name__ == '__main__':
    np.random.seed(2345)
    #--model files to run --
    #net_version, tag, fold, epoch
    #m_param = [(1,1,0,5),(1,1,4,5)];
    m_param = [(1,1,4,5)];
    model_weights = [1.0]*len(m_param);
    predict_fn = models.get_predict_function(m_param, model_weights);

    load_and_process = ld.LoadAndProcess(
            size = None,
            augmentation_params = None,
            crop=(cfg.WIDTH, cfg.HEIGHT),
            color_noise = 0);

    test_imgs,test_labels = ld.list_imgs_labels(os.path.join(cfg.dataset_dir,'test'));
    test_data = ld.ImgStream(test_imgs, test_labels, cfg.batch_size,
            istrain=False, file_dir_fmt=cfg.dataset_dir+'/test/{}',
            load_and_process = load_and_process);

    print("num of test cases: {}".format(len(test_data)));

    res = [];
    c = 0;
    for imgs,labels in test_data:
        res.append(predict_fn(imgs));
        c += 1;
        if c%10 == 0:
            print("{} processed ".format(c*32));

    res = np.concatenate(res);
    filename = cfg.output_dir + "/submit_v1.csv";
    print(res[-1])
    print(res[0])
    #utils.make_submission(filename, test_imgs, res, 0.5e-3);
