import cv2;
import numpy as np;
import os
import time

import multiprocessing as mp;
import Queue;

NUM_PROCESSES = 2

def list_imgs_labels(file_dir):
    imgs = [];
    labels = [];
    for f in os.listdir(file_dir):
        imgs.append(f);
        labels.append(int(f[:-5].split('_')[-1])-1);
    return imgs,labels

import img_augmentation as im_aug;
class LoadAndProcess(object):
    """
    a hack to make pool.map work with extra arguments by making a callable class
    """
    def __init__(self,size=None, augmentation_params=None, crop=None, color_noise=0):
        self.size = size;
        self.augmentation_params = augmentation_params;
        self.crop = crop;
        self.color_noise = color_noise;

    def __call__(self,filename):
        """
        size = (w,h)
        crop = (w,h)
        """
        img = cv2.imread(filename,3);
        if self.size is not None:
            img = cv2.resize(img, self.size); #size = W,H, img.shape=[H,w,c]

        img = img.astype(np.float32)*(2.0/255.0) - 1.0;
        img = img.transpose(2,1,0); # c, W, H

        if self.color_noise>0:
            img = np.clip(im_aug.augment_color(img, self.color_noise),-1,1);

        if self.augmentation_params is not None:
            img = im_aug.perturb(img, self.augmentation_params);

        c,W,H = img.shape;
        if self.crop is not None:
            x = np.random.randint(0,W-self.crop[0]);
            y = np.random.randint(0,H-self.crop[1]);
            img = img[:,x:x+self.crop[0],y:y+self.crop[1]];

        return img;

class ImgStream:
    def __init__(self, imgs, labels, batch_size, istrain=True, file_dir_fmt="{}", load_and_process = LoadAndProcess()):
        """
        For training, it will cycle the images again and again
        For testing/validating, notice that the last batch might smaller than batch_size
        """
        assert(len(imgs)==len(labels))
        self.imgs = imgs;
        self.labels = labels;
        self.batch_size = batch_size;
        self.istrain = istrain;
        self.size = len(self.imgs);
        self.Nbatch = self.size//self.batch_size;
        if self.Nbatch * self.batch_size < self.size:
            self.Nbatch += 1;
        self.file_dir_fmt = file_dir_fmt;
        self.load_and_process = load_and_process;
        self._concurrent = False

    def __len__(self):
        return self.size;

    def __iter__(self):
        if self._concurrent:
            return self._mp_datagen();
        else:
            return self._datagen();
    def _datagen(self): #single process
        idx = np.arange(self.size);
        while True:
            if self.istrain:
                np.random.shuffle(idx)
            for b in range(self.Nbatch):
                batch_idx = idx[b*self.batch_size:min((b+1)*self.batch_size,self.size)]
                if self.istrain and len(batch_idx)<self.batch_size:
                    continue
                yield self._load_data(batch_idx)

            if not self.istrain:
                break

    def _mp_datagen(self): #try to make it concurrent but did not work well :(
        databuffer = mp.Queue(maxsize=2);
        process = mp.Process(target=self._buffered_generation_process, args=(databuffer,));
        process.start()

        while True:
            try:
                try:
                    yield databuffer.get(True, timeout=1)
                except Queue.Empty:
                    if not process.is_alive():
                        break
                    pass
            except IOError:
                break

    def _buffered_generation_process(self,databuffer):
        idx = np.arange(self.size);
        while True:
            if self.istrain:
                np.random.shuffle(idx)
            for b in range(self.Nbatch):
                batch_idx = idx[b*self.batch_size:min((b+1)*self.batch_size,self.size)]
                if self.istrain and len(batch_idx)<self.batch_size:
                    continue
                while databuffer.full():
                    time.sleep(1)
                databuffer.put(self._load_data(batch_idx))

            if not self.istrain:
                databuffer.close()
                break

    def _load_data(self,batch_idx):
        pool = mp.Pool(NUM_PROCESSES);
        files = (self.file_dir_fmt.format(self.imgs[ii]) for ii in batch_idx);
        data_imgs = pool.map(self.load_and_process, files);
        pool.close()
        pool.join()
        data_imgs = np.asarray(data_imgs);
        data_labels = [self.labels[ii] for ii in batch_idx];
        data_labels = np.asarray(data_labels, np.int32);
        return data_imgs,data_labels;
