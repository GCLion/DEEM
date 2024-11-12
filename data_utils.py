import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
import random
import warnings
import numpy as np
import librosa
import scipy

def genSpoof_list( dir_meta,is_train=False,is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    count=0

    if (is_train):
        for line in l_meta:
             _, key,_,_,label = line.strip().split(' ')
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
             if label=='bonafide':
                 count=count+1
        print(count)
        print(len(d_meta))
        print(1-count *1.0/ len(d_meta))
        return d_meta,file_list

    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _, key,_,_,label = line.strip().split(' ')
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
             if label == 'bonafide':
                 count = count + 1
        print(count)
        print(len(d_meta))
        print(1-count*1.0 / len(d_meta))
        return d_meta,file_list


def genSpoof_list4ForTrain():
    d_meta = {}
    file_list = []

    for name in os.listdir('/save/for-norm/training/fake'):
        if os.path.isfile(os.path.join('/save/for-norm/training/fake', name)):
            file_list.append('training/fake/'+name)
            d_meta['training/fake/'+name]= 0

    for name in os.listdir('/save/for-norm/training/real'):
        if os.path.isfile(os.path.join('/save/for-norm/training/real', name)):
            file_list.append('training/real/'+name)
            d_meta['training/real/'+name]= 1
    return d_meta, file_list

def genSpoof_list4ForVaild():
    d_meta = {}
    file_list = []

    for name in os.listdir('/save/for-norm/validation/fake'):
        if os.path.isfile(os.path.join('/save/for-norm/validation/fake', name)):
            file_list.append('validation/fake/'+name)
            d_meta['validation/fake/'+name]= 0

    for name in os.listdir('/save/for-norm/validation/real'):
        if os.path.isfile(os.path.join('/save/for-norm/validation/real', name)):
            file_list.append('validation/real/'+name)
            d_meta['validation/real/'+name]= 1
    return d_meta, file_list

def genSpoof_list4ForTest():
    d_meta = {}
    file_list = []

    for name in os.listdir('/save/for-norm/testing/fake'):
        if os.path.isfile(os.path.join('/save/for-norm/testing/fake', name)):
            file_list.append('testing/fake/'+name)
            d_meta['testing/fake/'+name]= 0

    for name in os.listdir('/save/for-norm/testing/real'):
        if os.path.isfile(os.path.join('/save/for-norm/testing/real', name)):
            file_list.append('testing/real/'+name)
            d_meta['testing/real/'+name]= 1
    return d_meta, file_list



def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    count = 0

    if (is_train):
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
            if label == 'bonafide':
                count = count + 1
        print(count)
        print(len(d_meta))
        print(1 - count * 1.0 / len(d_meta))
        return d_meta, file_list

    elif (is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _,_, label,_,_ = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
            if label == 'bonafide':
                count = count + 1
        print(count)
        print(len(d_meta))
        print(1 - count * 1.0 / len(d_meta))
        return d_meta, file_list

def genSpoof_listwild(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    count = 0

    if (is_train):
        for line in l_meta:
            key,_, label = line.strip().split(',')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bona-fide' else 0
            if label == 'bona-fide':
                count = count + 1
        print(count)
        print(len(d_meta))
        print(1 - count * 1.0 / len(d_meta))
        return d_meta, file_list

    elif (is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            key, _, label = line.strip().split(',')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bona-fide' else 0
            if label == 'bona-fide':
                count = count + 1
        print(count)
        print(len(d_meta))
        print(1 - count * 1.0 / len(d_meta))
        return d_meta, file_list


def genSpoof_list_2019(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif (is_eval):
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


def genMydata(dir_meta):
    file_list1 = []
    file_list2 = []
    file_list3 = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()
    for line in l_meta:
        x, xs, xb = line.strip().split(' ')
        file_list1.append(x)
        file_list2.append(xs)
        file_list3.append(xb)
    c = list(zip(file_list1, file_list2,file_list3))
    random.shuffle(c)
    file_list1[:], file_list2[:] ,file_list3[:]= zip(*c)

    return file_list1, file_list2,file_list3

def genMydata2(dir_meta,a,b):
    file_list_a = []
    file_list_b = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        _, key, _, _, label = line.strip().split(' ')
        if label=='bonafide':
            file_list_a.append(key)
        else:
            file_list_b.append(key)
    random.shuffle(file_list_a)
    random.shuffle(file_list_b)
    re=file_list_a[:a]+file_list_b[:b]
    random.shuffle(re)
    print(len(re))
    return re

def genMydata4For(a,b):
    file_list_a = ['real/'+name for name in os.listdir('/save/for-norm/training/real')
                if os.path.isfile(os.path.join('/save/for-norm/training/real', name))]
    file_list_b = ['fake/'+name for name in os.listdir('/save/for-norm/training/fake')
                if os.path.isfile(os.path.join('/save/for-norm/training/fake', name))]

    random.shuffle(file_list_a)
    random.shuffle(file_list_b)
    re=file_list_a[:a]+file_list_b[:b]
    random.shuffle(re)
    print(len(re))
    return re

def genMydata3(dir_meta,a,b):
    file_list_a = []
    file_list_b = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        _, key, _, _, _, label,_,_ = line.strip().split(' ')
        if label=='bonafide':
            file_list_a.append(key)
        else:
            file_list_b.append(key)
    random.shuffle(file_list_a)
    random.shuffle(file_list_b)
    re=file_list_a[:a]+file_list_b[:b]
    random.shuffle(re)
    print(len(re))
    return re

def pad(x,max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
            

    def __len__(self):
           return len(self.list_IDs)


    def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X,fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000)
            X_pad= pad(X,self.cut)
            x_inp= Tensor(X_pad)
            y = self.labels[key]
            return x_inp, y


class Dataset_ASVspoof2019_train_For(Dataset):
    def __init__(self, list_IDs, labels):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        X, fs = librosa.load('/save/for-norm/' + key, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + key + '.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return key,x_inp, y

class Dataset_ASVspoofwild_train(Dataset):
    def __init__(self, list_IDs, labels):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        X, fs = librosa.load('/save/wild/release_in_the_wild/' + key, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_train_LFCC(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + key + '.flac', sr=16000)

        X_pad = pad(X, self.cut)
        x_lfcc = lfcc(y=X_pad, sr=fs, n_lfcc=128, n_fft=512, hop_length=239, n_filters=128, pad_mode='reflect')
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_lfcc,x_inp, y
            
            
class Dataset_ASVspoof_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''

            self.list_IDs = list_IDs
            self.base_dir = base_dir
            

    def __len__(self):
            return len(self.list_IDs)


    def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000)
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,key


class Dataset_ae(Dataset):
    def __init__(self, listx,listxs,listxb,listy,base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.listx = listx
        self.listxs = listxs
        self.listxb = listxb
        self.listy = listy
        self.base_dir = base_dir
        self.cut=64600

    def __len__(self):
        # return len(self.listx)
        return 5000;
    def __getitem__(self, index):
        keyx1 = self.listx[index*2]
        keyxs1 = self.listxs[index*2]
        keyxb1 = self.listxb[index*2]

        keyx2 = self.listx[index*2+1]
        keyxs2 = self.listxs[index*2+1]
        keyxb2 = self.listxb[index*2+1]

        keyx1_=keyxs1+"_"+keyxb2
        keyx2_=keyxs2+"_"+keyxb1

        x1, fs = librosa.load(self.base_dir+'/together/'+keyx1+'.wav', sr=16000)
        x1_pad = pad(x1,self.cut)
        x1_inp = Tensor(x1_pad)

        x2, fs = librosa.load(self.base_dir+'/together/'+keyx2+'.wav', sr=16000)
        x2_pad = pad(x2,self.cut)
        x2_inp = Tensor(x2_pad)

        x1_, fs = librosa.load(self.base_dir+'/together/'+keyx1_+'.wav', sr=16000)
        x1__pad = pad(x1_,self.cut)
        x1__inp = Tensor(x1__pad)

        x2_, fs = librosa.load(self.base_dir+'/together/'+keyx2_+'.wav', sr=16000)
        x2__pad = pad(x2_,self.cut)
        x2__inp = Tensor(x2__pad)
        keyy1=self.listy[index*2]
        keyy2=self.listy[index*2+1]

        y1, fs = librosa.load(self.base_dir+'/flac/'+keyy1+'.flac', sr=16000)
        y1_pad = pad(y1,self.cut)
        y1_inp = Tensor(y1_pad)

        y2, fs = librosa.load(self.base_dir+'/flac/'+keyy2+'.flac', sr=16000)
        y2_pad = pad(y2,self.cut)
        y2_inp = Tensor(y2_pad)

        return x1_inp,x2_inp,x1__inp,x2__inp,y1_inp,y2_inp

class Dataset_ae4For(Dataset):
    def __init__(self, listx,listxs,listxb,listy,base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.listx = listx
        self.listxs = listxs
        self.listxb = listxb
        self.listy = listy
        self.base_dir = base_dir
        self.cut=64600

    def __len__(self):
        # return len(self.listx)
        return 5000;
    def __getitem__(self, index):
        keyx1 = self.listx[index*2]
        keyxs1 = self.listxs[index*2]
        keyxb1 = self.listxb[index*2]

        keyx2 = self.listx[index*2+1]
        keyxs2 = self.listxs[index*2+1]
        keyxb2 = self.listxb[index*2+1]

        keyx1_=keyxs1+"_"+keyxb2
        keyx2_=keyxs2+"_"+keyxb1

        x1, fs = librosa.load(self.base_dir+'/together/'+keyx1+'.wav', sr=16000)
        x1_pad = pad(x1,self.cut)
        x1_inp = Tensor(x1_pad)

        x2, fs = librosa.load(self.base_dir+'/together/'+keyx2+'.wav', sr=16000)
        x2_pad = pad(x2,self.cut)
        x2_inp = Tensor(x2_pad)

        x1_, fs = librosa.load(self.base_dir+'/together/'+keyx1_+'.wav', sr=16000)
        x1__pad = pad(x1_,self.cut)
        x1__inp = Tensor(x1__pad)

        x2_, fs = librosa.load(self.base_dir+'/together/'+keyx2_+'.wav', sr=16000)
        x2__pad = pad(x2_,self.cut)
        x2__inp = Tensor(x2__pad)
        keyy1=self.listy[index*2]
        keyy2=self.listy[index*2+1]

        y1, fs = librosa.load('/save/for-norm/training/'+keyy1, sr=16000)
        y1_pad = pad(y1,self.cut)
        y1_inp = Tensor(y1_pad)

        y2, fs = librosa.load('/save/for-norm/training/'+keyy2, sr=16000)
        y2_pad = pad(y2,self.cut)
        y2_inp = Tensor(y2_pad)

        return x1_inp,x2_inp,x1__inp,x2__inp,y1_inp,y2_inp

class Dataset_ae_finetune(Dataset):
    def __init__(self, listx,listxs,listxb,listy,base_dir,base_dir1):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.listx = listx
        self.listxs = listxs
        self.listxb = listxb
        self.listy = listy
        self.base_dir = base_dir
        self.base_dir1=base_dir1
        self.cut=64600

    def __len__(self):
        # return len(self.listx)
        return 5000;
    def __getitem__(self, index):
        keyx1 = self.listx[index*2]
        keyxs1 = self.listxs[index*2]
        keyxb1 = self.listxb[index*2]

        keyx2 = self.listx[index*2+1]
        keyxs2 = self.listxs[index*2+1]
        keyxb2 = self.listxb[index*2+1]

        keyx1_=keyxs1+"_"+keyxb2
        keyx2_=keyxs2+"_"+keyxb1

        x1, fs = librosa.load(self.base_dir+'/together/'+keyx1+'.wav', sr=16000)
        x1_pad = pad(x1,self.cut)
        x1_inp = Tensor(x1_pad)

        x2, fs = librosa.load(self.base_dir+'/together/'+keyx2+'.wav', sr=16000)
        x2_pad = pad(x2,self.cut)
        x2_inp = Tensor(x2_pad)

        x1_, fs = librosa.load(self.base_dir+'/together/'+keyx1_+'.wav', sr=16000)
        x1__pad = pad(x1_,self.cut)
        x1__inp = Tensor(x1__pad)

        x2_, fs = librosa.load(self.base_dir+'/together/'+keyx2_+'.wav', sr=16000)
        x2__pad = pad(x2_,self.cut)
        x2__inp = Tensor(x2__pad)
        keyy1=self.listy[index*2]
        keyy2=self.listy[index*2+1]

        y1, fs = librosa.load(self.base_dir1+'/flac/'+keyy1+'.flac', sr=16000)
        y1_pad = pad(y1,self.cut)
        y1_inp = Tensor(y1_pad)

        y2, fs = librosa.load(self.base_dir1+'/flac/'+keyy2+'.flac', sr=16000)
        y2_pad = pad(y2,self.cut)
        y2_inp = Tensor(y2_pad)

        return x1_inp,x2_inp,x1__inp,x2__inp,y1_inp,y2_inp



def linear(sr, n_fft, n_filters=128, fmin=0.0, fmax=None, dtype=np.float32):

    if fmax is None:
        fmax = float(sr) / 2
    # Initialize the weights
    n_filters = int(n_filters)
    weights = np.zeros((n_filters, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of liner bands - uniformly spaced between limits
    # * Need to validate
    linear_f = np.linspace(fmin, fmax, n_filters + 2)

    fdiff = np.diff(linear_f)
    ramps = np.subtract.outer(linear_f, fftfreqs)

    for i in range(n_filters):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Only check weights if f_mel[0] is positive
    if not np.all((linear_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in linear frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_filters.",
            stacklevel=2,
        )

    return weights


def linear_spec(
    y=None,
        sr=22050,
        S=None,
        n_fft=2048,
        hop_length=512,
        win_length=None,
        window='hann',
        center=True,
        pad_mode='constant',
        power=2.0,
        **kwargs
):
    if S is not None:
        # Infer n_fft from spectrogram shape, but only if it mismatches
        if n_fft // 2 + 1 != S.shape[-2]:
            n_fft = 2 * (S.shape[-2] - 1)
    else:
        S = (
            np.abs(
                librosa.stft(
                    y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    window=window,
                    pad_mode=pad_mode,
                )
            )
            ** power
        )
    # Build a linear filter
    linear_basis = linear(sr=sr, n_fft=n_fft, **kwargs)

    return np.einsum("...ft,mf->...mt", S, linear_basis, optimize=True)


def expand_to(x, *, ndim, axes):
    try:
        axes = tuple(axes)
    except TypeError:
        axes = tuple([axes])

    if len(axes) != x.ndim:
        raise Exception(
            "Shape mismatch between axes={} and input x.shape={}".format(
                axes, x.shape)
        )

    if ndim < x.ndim:
        raise Exception(
            "Cannot expand x.shape={} to fewer dimensions ndim={}".format(
                x.shape, ndim)
        )

    shape = [1] * ndim
    for i, axi in enumerate(axes):
        shape[axi] = x.shape[i]

    return x.reshape(shape)


def lfcc(y=None,
         sr=22050,
         S=None,
         n_lfcc=20,
         dct_type=2,
         norm='ortho',
         lifter=0,
         **kwargs):
    if S is None:
        S = librosa.power_to_db(linear_spec(y=y, sr=sr, **kwargs))

    M = scipy.fftpack.dct(S, axis=-2, type=dct_type,
                          norm=norm)[..., :n_lfcc, :]

    if lifter > 0:
        # shape lifter for broadcasting
        LI = np.sin(np.pi * np.arange(1, 1 + n_lfcc, dtype=M.dtype) / lifter)
        LI = expand_to(LI, ndim=S.ndim, axes=-2)

        M *= 1 + (lifter / 2) * LI
        return M
    elif lifter == 0:
        return M
    else:
        raise Exception(
            "LFCC lifter={} must be a non-negative number".format(lifter)
        )


class Dataset_ae_lfcc(Dataset):
    def __init__(self, listx,listxs,listxb,listy,base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.listx = listx
        self.listxs = listxs
        self.listxb = listxb
        self.listy = listy
        self.base_dir = base_dir
        self.cut=64600

    def __len__(self):
        return 5000;
    def __getitem__(self, index):
        keyx1 = self.listx[index*2]
        keyxs1 = self.listxs[index*2]
        keyxb1 = self.listxb[index*2]

        keyx2 = self.listx[index*2+1]
        keyxs2 = self.listxs[index*2+1]
        keyxb2 = self.listxb[index*2+1]

        keyx1_=keyxs1+"_"+keyxb2
        keyx2_=keyxs2+"_"+keyxb1

        x1, fs = librosa.load(self.base_dir+'/together/'+keyx1+'.wav', sr=16000)
        x1_pad = pad(x1,self.cut)
        x1_pad = lfcc(y=x1_pad, sr=fs, n_lfcc=64, n_fft=512, hop_length=239, n_filters=128, pad_mode='reflect')
        x1_inp = Tensor(x1_pad)

        x2, fs = librosa.load(self.base_dir+'/together/'+keyx2+'.wav', sr=16000)
        x2_pad = pad(x2,self.cut)
        x2_pad = lfcc(y=x2_pad, sr=fs, n_lfcc=64, n_fft=512, hop_length=239, n_filters=128, pad_mode='reflect')
        x2_inp = Tensor(x2_pad)

        x1_, fs = librosa.load(self.base_dir+'/together/'+keyx1_+'.wav', sr=16000)
        x1__pad = pad(x1_,self.cut)
        x1__pad = lfcc(y=x1__pad, sr=fs, n_lfcc=64, n_fft=512, hop_length=239, n_filters=128, pad_mode='reflect')
        x1__inp = Tensor(x1__pad)

        x2_, fs = librosa.load(self.base_dir+'/together/'+keyx2_+'.wav', sr=16000)
        x2__pad = pad(x2_,self.cut)
        x2__pad = lfcc(y=x2__pad, sr=fs, n_lfcc=64, n_fft=512, hop_length=239, n_filters=128, pad_mode='reflect')
        x2__inp = Tensor(x2__pad)

        keyy1=self.listy[index*2]
        keyy2=self.listy[index*2+1]

        y1, fs = librosa.load(self.base_dir+'/flac/'+keyy1+'.flac', sr=16000)
        y1_pad = pad(y1,self.cut)
        y1_pad = lfcc(y=y1_pad, sr=fs, n_lfcc=64, n_fft=512, hop_length=239, n_filters=128, pad_mode='reflect')
        y1_inp = Tensor(y1_pad)

        y2, fs = librosa.load(self.base_dir+'/flac/'+keyy2+'.flac', sr=16000)
        y2_pad = pad(y2,self.cut)
        y2_pad = lfcc(y=y2_pad, sr=fs, n_lfcc=64, n_fft=512, hop_length=239, n_filters=128, pad_mode='reflect')
        y2_inp = Tensor(y2_pad)

        return x1_inp,x2_inp,x1__inp,x2__inp,y1_inp,y2_inp

class Dataset_ASVspoof2019_train_lfcc(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + key + '.flac', sr=16000)
        X_pad = pad(X, self.cut)
        X_pad = lfcc(y=X_pad, sr=fs, n_lfcc=64, n_fft=512, hop_length=240, n_filters=128, pad_mode='reflect')
        x_inp = Tensor(X_pad.flatten())
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof_eval_lfcc(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + key + '.flac', sr=16000)
        X_pad = pad(X, self.cut)
        X_pad = lfcc(y=X_pad, sr=fs, n_lfcc=64, n_fft=512, hop_length=240, n_filters=128, pad_mode='reflect')
        x_inp = Tensor(X_pad.flatten())
        return x_inp, key



