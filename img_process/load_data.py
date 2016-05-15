import cv2;
import numpy as np;
import os
import time

import multiprocessing as mp;
import Queue;

NUM_PROCESSES = 3

##customerize this function to find the images
def list_imgs_labels(file_dir, data): #data = train or test
    imgs = [];
    labels = [];
    file_dir = os.path.join(file_dir, data);

    #for f in os.listdir(file_dir):
    #    imgs.append(f);
    #    labels.append(int(f[:-5].split('_')[-1])-1);

    if data=='train':
        for l in range(10):
            label = 'c'+str(l)
            l_dir = os.path.join(file_dir,label)
            for f in os.listdir(l_dir):
                imgs.append(os.path.join(label,f))
                labels.append(l)
    else:
        for f in os.listdir(file_dir):
            imgs.append(f)
            labels.append(0)

    return imgs,labels

import img_augmentation as im_aug;
class LoadAndProcess(object):
    """
    a hack to make pool.map work with extra arguments by making a callable class
    """
    def __init__(self,size=None, augmentation_params=None, crop=None, color_noise=0, fill_size=None):
        """
        size = (w,h)
        crop = (w,h)
        """
        self.size = size;
        self.augmentation_params = augmentation_params;
        self.crop = crop;
        self.color_noise = color_noise;
        self.fill_size = fill_size; 

    def load(self,filename):
        img = cv2.imread(filename,3);
        if self.size is not None:
            img = cv2.resize(img, self.size); #size = W,H, img.shape=[H,w,c]
        img = img.transpose(2,1,0); # c, W, H
        return img

    def __call__(self,img): #augmentation
        img = img.astype(np.float32)*(2.0/255.0) - 1.0;
        if self.color_noise>0:
            img = np.clip(im_aug.augment_color(img, self.color_noise),-1,1);

        if self.fill_size is not None and self.fill_size>0:
            img = im_aug.match_pretrained(img, self.fill_size)

        if self.augmentation_params is not None:
            img = im_aug.perturb(img, self.augmentation_params);

        c,W,H = img.shape;
        if self.crop is not None:
            x = np.random.randint(0,W-self.crop[0]);
            y = np.random.randint(0,H-self.crop[1]);
            img = img[:,x:x+self.crop[0],y:y+self.crop[1]];

        return img

class ImgStream:
    def __init__(self, imgs, labels, batch_size, cycle, file_dir_fmt, load_and_process = LoadAndProcess(), preload=None):
        """
        For training, it will cycle the images again and again
        For testing/validating, notice that the last batch might smaller than batch_size
        preload provides file directory, if file exists, then simply load it, or else generate it and save it
        """
        assert(len(imgs)==len(labels))
        self.imgs = imgs;
        self.labels = labels;
        self.batch_size = batch_size;
        self.cycle = cycle;
        self.size = len(self.imgs);
        self.file_dir_fmt = file_dir_fmt;
        self.load_and_process = load_and_process;
        self._concurrent = False #does not seem to work

        self.img_data  = None;
        if preload is not None:
            if os.path.isfile(preload):
                print("load from saved npy file: {}".format(preload));
                self.img_data = np.load(preload);
            else:
                self.img_data = [];
                print("preload all the images")
                for name in self.imgs:
                    self.img_data.append(self.load_and_process.load(self.file_dir_fmt.format(name)));
                self.img_data = np.asarray(self.img_data);
                print("save to npy file: {}".format(preload));
                np.save(preload,self.img_data);

    def __len__(self):
        return self.size;

    def __iter__(self):
        if self._concurrent:
            raise NotImplementedError
            #return self._mp_datagen();
        else:
            return self._datagen();

    def CV_fold(self, fold, NCV, istrain, split_id=None): #istrain=False means validation set
        """
        returns a generator that iterate the expected set
        """
        idx = np.arange(self.size);
        if fold==NCV:
            if not istrain:
                return None
        elif fold<NCV:
            if split_id is not None:
                iii = split_id%NCV == fold;
            else:
                iii = idx%NCV == fold;
            if istrain:
                idx = idx[~iii];
            else:
                idx = idx[iii];
        else:
            raise LookupError
        if istrain:
            return self._datagen(idx, cycle=True);
        else:
            return self._datagen(idx, cycle=False);

    def _datagen(self,idx=None,cycle=None): #single process
        if idx is None:
            idx = np.arange(self.size);
        if cycle is None:
            cycle = self.cycle;
        L = len(idx)
        Nbatch = L//self.batch_size;
        if Nbatch * self.batch_size < L:
            Nbatch += 1;
        while True:
            if cycle:
                np.random.shuffle(idx)
            for b in range(Nbatch):
                batch_idx = idx[b*self.batch_size:min((b+1)*self.batch_size,L)]
                if cycle and len(batch_idx)<self.batch_size:
                    continue
                yield self._load_data(batch_idx)

            if not cycle:
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
        files = (self.file_dir_fmt.format(self.imgs[ii]) for ii in batch_idx);
        if self.img_data is None:
            imgs = [self.load_and_process.load(f) for f in files]
        else:
            imgs = self.img_data[batch_idx];

        #pool = mp.Pool(NUM_PROCESSES);
        #data_imgs = pool.map(self.load_and_process, imgs);
        #pool.close()
        #pool.join()
        data_imgs = [self.load_and_process(im) for im in imgs];

        data_imgs = np.asarray(data_imgs);
        data_labels = [self.labels[ii] for ii in batch_idx];
        data_labels = np.asarray(data_labels, np.int32);
        return data_imgs,data_labels;
