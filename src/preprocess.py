# Reference: https://groups.google.com/forum/#!msg/librosa/V4Z1HpTKn8Q/1-sMpjxjCSoJ

import librosa
import numpy as np
from operator import itemgetter
# NOTE: there are warnings for MFCC extraction due to librosa's issue
import warnings
warnings.filterwarnings("ignore")

import torch
import random
import matplotlib.pyplot as plt
from exp.nb_SparseImageWarp import sparse_image_warp

import os
from scipy import misc
from PIL import Image

# Acoustic Feature Extraction
# Parameters
#     - input file  : str, audio file path
#     - feature     : str, fbank or mfcc
#     - dim         : int, dimension of feature
#     - cmvn        : bool, apply CMVN on feature
#     - window_size : int, window size for FFT (ms)
#     - stride      : int, window stride for FFT
#     - save_feature: str, if given, store feature to the path and return len(feature)
# Return
#     acoustic features with shape (time step, dim)
def extract_feature(input_file,feature='fbank',dim=40, cmvn=True, delta=False, delta_delta=False,
                    window_size=25, stride=10,spec_aug=False, save_spec=False, save_feature=None):
    y, sr = librosa.load(input_file,sr=None)
    ws = int(sr*0.001*window_size)
    st = int(sr*0.001*stride)
    if feature == 'fbank': # log-scaled
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim,
                                    n_fft=ws, hop_length=st)
        feat = np.log(feat+1e-6)
    elif feature == 'mfcc':
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim, n_mels=26,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rmse(y, hop_length=st, frame_length=ws)

    else:
        raise ValueError('Unsupported Acoustic Feature: '+feature)

    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta:
        feat.append(librosa.feature.delta(feat[0],order=2))
    feat = np.concatenate(feat,axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:,np.newaxis]) / (feat.std(axis=1)+1e-16)[:,np.newaxis]

    # if spec_aug:
    #     # tensor_to_img(feat,input_file.split('/')[-1].replace('.wav',''))
    #     # print('spec_aug = True')
    #     feat = torch.from_numpy(feat) # ndarry to tensor
    #     feat = time_warp(feat.view(1, feat.shape[0], feat.shape[1]))
    #     feat = freq_mask(feat)
    #     feat = time_mask(feat)
    #     feat = feat.numpy() # tensor to ndarry
    #     # tensor_to_img(feat,input_file.split('/')[-1].replace('.wav','')+'_aug')

    # if save_spec:
    #     array_to_img(feat, input_file.split('/')[-1].replace('.wav', ''))
    #     # print(feat.shape,feat.dtype)

    if save_feature is not None:
        tmp = np.swapaxes(feat,0,1).astype('float32')
        np.save(save_feature,tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat,0,1).astype('float32')

# Target Encoding Function
# Parameters
#     - input list : list, list of target list
#     - table      : dict, token-index table for encoding (generate one if it's None)
#     - mode       : int, encoding mode ( phoneme / char / subword / word )
#     - max idx    : int, max encoding index (0=<sos>, 1=<eos>, 2=<unk>)
# Return
#     - output list: list, list of encoded targets
#     - output dic : dict, token-index table used during encoding
def encode_target(input_list,table=None,mode='subword',max_idx=500):
    if table is None:
        ### Step 1. Calculate word frequency
        table = {}
        for target in input_list:
            for t in target:
                if t not in table:
                    table[t] = 1
                else:
                    table[t] += 1
        ### Step 2. Top k list for encode map
        max_idx = min(max_idx-3,len(table))
        all_tokens = [k for k,v in sorted(table.items(), key = itemgetter(1), reverse = True)][:max_idx]
        table = {'<sos>':0,'<eos>':1}
        if mode == "word": table['<unk>']=2
        for tok in all_tokens:
            table[tok] = len(table)
    ### Step 3. Encode
    output_list = []
    for target in input_list:
        tmp = [0]
        for t in target:
            if t in table:
                tmp.append(table[t])
            else:
                if mode == "word":
                    tmp.append(2)
                else:
                    tmp.append(table['<unk>'])
                    # raise ValueError('OOV error: '+t)
        tmp.append(1)
        output_list.append(tmp)
    return output_list,table


# Feature Padding Function 
# Parameters
#     - x          : list, list of np.array
#     - pad_len    : int, length to pad (0 for max_len in x)      
# Return
#     - new_x      : np.array with shape (len(x),pad_len,dim of feature)
def zero_padding(x,pad_len):
    features = x[0].shape[-1]
    if pad_len is 0: pad_len = max([len(v) for v in x])
    new_x = np.zeros((len(x),pad_len,features))
    for idx,ins in enumerate(x):
        new_x[idx,:min(len(ins),pad_len),:] = ins[:min(len(ins),pad_len),:]
    return new_x

# Target Padding Function 
# Parameters
#     - y          : list, list of int
#     - max_len    : int, max length of output (0 for max_len in y)     
# Return
#     - new_y      : np.array with shape (len(y),max_len)
def target_padding(y,max_len):
    if max_len is 0: max_len = max([len(v) for v in y])
    new_y = np.zeros((len(y),max_len),dtype=int)
    for idx,label_seq in enumerate(y):
        new_y[idx,:len(label_seq)] = np.array(label_seq)
    return new_y


# Leslie add
# For Spec Augment
def time_warp(spec, W=3):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]
    device = spec.device

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = (torch.tensor([[[y, point_to_warp]]], device=device),
                         torch.tensor([[[y, point_to_warp + dist_to_warp]]], device=device))
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3).squeeze(0)


def freq_mask(spec, F=20, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[0]

    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)

        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned

        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero):
            cloned[f_zero:mask_end] = 0
        else:
            cloned[f_zero:mask_end] = cloned.mean()

    return cloned


def time_mask(spec, T=40, num_masks=1, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[1]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)

        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned

        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero):
            cloned[:, t_zero:mask_end] = 0
        else:
            cloned[:, t_zero:mask_end] = cloned.mean()
    return cloned


def tensor_to_img(spectrogram, name):
    path = os.path.join('/media/ee303/Transcend/new_test/aug_picture/spec',name+'.jpg')
    plt.figure() # arbitrary,
    plt.imshow(spectrogram)
    plt.savefig(path)
    # plt.show()


def array_to_img(spectrogram, name):
    path = os.path.join('/home/ee303/Documents/LAS/End-to-end-ASR-Pytorch/pic/real_A_train',name+'.jpg')
    misc.imsave(path, spectrogram)
    # plt.show()


def spec_feature(input_file,save_feature=None):
    img = Image.open(input_file)
    data = np.array(img)

    if save_feature is not None:
        tmp = np.swapaxes(data,0,1).astype('float32')
        print(tmp,tmp.shape, tmp.dtype)
        np.save(save_feature,tmp)
        return len(tmp)
