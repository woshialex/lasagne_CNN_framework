import sys
from SETTINGS import *;
tag = 1;
net_version = 1;
WIDTH=256;
HEIGHT=WIDTH;
batch_size = 32;
num_epochs = 5;
save_epoch = [];

#CV = [(0,4),(4,4)]; #CV folds, (N,N) means uses all data
CV = [(4,4)];

#augmentation parameters
#shift = 35;
#rotation = 10;
#shear = 8;
#scale = 0.05;
#color_noise = 0.2;

learn_rate = 1.0e-3;
lr_decay = {2:0.2e-3, 70:5.0e-4}
