import sys
from SETTINGS import *;
tag = 1;

net_version = 8;
pretrained = 299;#pretrained size

WIDTH=192;
HEIGHT=WIDTH*480/640;
batch_size = 8;
num_epochs = 30;
save_epoch = [10,20];

NCV = 4;
folds = [4]; #fold == NCV means to use all data

#augmentation parameters
shift = 20;
rotation = 10;
shear = 8;
scale = 0.06;
color_noise = 0.7;
crop = None

learn_rate = 0.5e-3;
#lr_decay = {10:2e-3, 15:0.5e-3}
lr_decay = {}
