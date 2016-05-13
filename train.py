#!/usr/bin/python2
import theano
import theano.tensor as T
import lasagne as nn

import os
from CNN_models.models import *

import numpy as np
import config as cfg
from img_process import load_data as ld
from sklearn.cross_validation import KFold
import time

if __name__=='__main__':
    """
    build and train the CNNs.
    """
    np.random.seed(1234);

    #aug_params = {
    #    'zoom_range': (1/(1+cfg.scale), 1+cfg.scale),
    #    'rotation_range': (-cfg.rotation, cfg.rotation),
    #    'shear_range': (-cfg.shear, cfg.shear),
    #    'translation_range': (-cfg.shift, cfg.shift),
    #    'do_flip': False,
    #    'allow_stretch': True,
    #}
    aug_params = None;
    #image augmentation
    load_and_process = ld.LoadAndProcess(
            size = None,
            augmentation_params = aug_params,
            crop=(cfg.WIDTH, cfg.HEIGHT),
            color_noise = 0);

    input_var = T.tensor4('input')
    label_var = T.ivector('label')

    net, output, output_det = build_cnn(input_var, (None, 3, cfg.WIDTH, cfg.HEIGHT), 
            version=cfg.net_version)
    ###continue training !!!!
    #u.load_params(net['output'], cfg.params_dir + '/cnn_v7_tag14_f4.npz');

    for l in nn.layers.get_all_layers(net['output']):
        print nn.layers.get_output_shape(l)
    params = nn.layers.get_all_params(net['output'], trainable=True)
    init0 = nn.layers.get_all_param_values(net['output']);
    
    lr = theano.shared(nn.utils.floatX(cfg.learn_rate))
    penalty = nn.regularization.regularize_network_params(net['output'],nn.regularization.l2);
    l2_lambda = 1e-6; 
    loss = nn.objectives.categorical_crossentropy(output, label_var).mean() + penalty*l2_lambda;

    cap = 1e-4;
    te_loss = nn.objectives.categorical_crossentropy(T.clip(output_det,cap,1-cap),label_var).mean();
    #te_loss = nn.objectives.categorical_crossentropy(output_det,label_var).mean();
    #te_acc = nn.objectives.categorical_accuracy(output_det, label_var).mean()

    updates = nn.updates.adam(loss, params, learning_rate=lr);#,beta1=0.95,beta2=0.998)
    #updates = nn.updates.nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9);

    train_fn = theano.function([input_var, label_var], loss, updates=updates)
    test_fn = theano.function([input_var, label_var], te_loss)
    #acc_fn = theano.function([input_var, label_var], te_acc)
    pred_fn = theano.function([input_var], output_det)

    #load data
    train_imgs, train_labels = ld.list_imgs_labels(os.path.join(cfg.dataset_dir,'train'));

    best_epoch = 0;
    for fold,NCV in cfg.CV:
        nn.layers.set_all_param_values(net['output'],init0);

        if fold == NCV:
            num_epochs = best_epoch+1 if best_epoch>0 else cfg.num_epochs;
            print("full train data use {:d} epochs".format(num_epochs))
            train_data = ld.ImgStream(train_imgs, train_labels, cfg.batch_size,
                    istrain=True, file_dir_fmt=cfg.dataset_dir+'/train/{}', 
                    load_and_process=load_and_process);
            val_data = None;
        else:
            num_epochs = cfg.num_epochs;
            train_data = ld.ImgStream([x for i,x in enumerate(train_imgs) if i%NCV!=fold],
                    [x for i,x in enumerate(train_labels) if i%NCV!=fold],
                    cfg.batch_size, istrain=True, file_dir_fmt=cfg.dataset_dir+'/train/{}',
                    load_and_process=load_and_process);
            val_data = ld.ImgStream(train_imgs[::NCV],train_labels[::NCV],cfg.batch_size,
                    istrain=False, file_dir_fmt=cfg.dataset_dir+'/train/{}',
                    load_and_process=load_and_process);

        #do training and validation
        epoch = 0;
        batches = 0;
        tr_err = 0.0;
        NBatch = len(train_data)//cfg.batch_size;
        start = time.time();
        best_val_err = 1.0e20;
        for imgs,labels in train_data:
            tr_err += train_fn(imgs,labels);
            batches += 1;
            if batches%NBatch == 0: #a new epoch
                epoch += 1;
                tr_err /= NBatch;

                # check validation results
                val_err = 0.0;
                if val_data is not None:
                    for val_imgs,val_labels in val_data:
                        val_err += test_fn(val_imgs, val_labels)*len(val_imgs);
                    val_err /= len(val_data);
                    if val_err<best_val_err:
                        best_val_err = val_err;
                        best_epoch = max(best_epoch,epoch);

                print('epoch {}/{} - tl {:.5f} - vl {:.5f} - t {:.3f}s'.format(
                    epoch, num_epochs, tr_err, val_err, time.time()-start))
                start = time.time();
                tr_err = 0.0;

                if epoch in cfg.lr_decay:
                    lr.set_value(cfg.lr_decay[epoch]);
                    print("learn rate:",lr.get_value());

                if epoch >= num_epochs:
                    break

                if epoch in cfg.save_epoch:
                    np.savez(cfg.params_dir + '/cnn{}_tag{}_f{}_ep{}.npz'.format(cfg.net_version,cfg.tag,fold,epoch), *nn.layers.get_all_param_values(net['output']))

        #save the trained model
        np.savez(cfg.params_dir + '/cnn{}_tag{}_f{}_ep{}.npz'.format(cfg.net_version,cfg.tag,fold,epoch), *nn.layers.get_all_param_values(net['output']))
