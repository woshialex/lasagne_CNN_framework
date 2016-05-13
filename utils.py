import scipy
import numpy as np
import lasagne as nn
import cv2
import config as cfg;

# loads params in npz
def load_params(model, fn):
    with np.load(fn) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    nn.layers.set_all_param_values(model, param_values)

def cross_entropy(pred, label, cap = 1e-5):
    x = np.clip(pred,cap,1-cap);
    x = x/np.sum(x,axis=1)[:,np.newaxis];
    y = np.log(x);
    return np.mean([y[i,int(label[i])] for i in range(len(label))]);

def binary_entropy(pred, label, cap = 1e-5):
    x = np.clip(pred,cap,1-cap);
    return np.mean(np.log(x)*label + np.log(1-x)*(1-label));

def accuracy(pred, label):
    m = [int(label[i]) == np.argmax(pred[i]) for i in range(len(label))];
    return 1.0*np.sum(m)/len(m);

def report(pred,label):
    print("mean prediction");
    for i in range(10):
        mp = np.mean(pred[label==i],axis=0);
        print("{} {}".format(i,' '.join(['{:.2f}'.format(x) for x in mp])));

def report_driver(pred, label, driver):
    m = np.asarray([int(label[i]) == np.argmax(pred[i]) for i in range(len(label))]);
    print("accuracy by driver : ");
    for i in np.unique(driver):
        idx = (driver == i);
        print("driver {}: {}".format(i,1.0*np.sum(m[idx])/np.sum(idx)));
    print("mean prediction by driver");
    for i in np.unique(driver):
        idx = (driver == i);
        mp = np.mean(pred[idx],axis=0);
        print("{} {}".format(i,' '.join(['{:.2f}'.format(x) for x in mp])));

def make_submission(filename, name_idx, res, cap = 1e-5):
    submit_csv = open(filename,'w');
    submit_csv.write("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n");
    res = np.clip(res,cap,1-cap);
    for i in range(len(name_idx)):
        submit_csv.write("img_{}.jpg,".format(name_idx[i]));
        submit_csv.write(','.join([str(x) for x in res[i]]));
        submit_csv.write('\n');
    submit_csv.close();

def save_cv(filename,res,label,ids):
    submit_csv = open(filename,'w');
    submit_csv.write("id,label,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n");
    for i in range(len(label)):
        submit_csv.write("{},{},".format(ids[i],label[i]));
        submit_csv.write(','.join([str(x) for x in res[i]]));
        submit_csv.write('\n');
    submit_csv.close();

def save_cv_cat(filename,res,label,ids,cat):
    submit_csv = open(filename,'w');
    submit_csv.write("id,label,c{}\n".format(cat));
    for i in range(len(label)):
        submit_csv.write("{},{},{}\n".format(ids[i],label[i],res[i]));
    submit_csv.close();
