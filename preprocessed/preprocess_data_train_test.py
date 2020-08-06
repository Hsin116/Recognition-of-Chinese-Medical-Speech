'''
For train/test preprocessing.
'''

import sys
sys.path.insert(0, '..')
from src.preprocess import extract_feature,encode_target
import argparse
import os
import pickle
import pandas as pd
import json


def boolean_string(s):
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser(description='Preprocess program for dataset.')
parser.add_argument('--data_path', default="/home/ee303/Documents/LAS/End-to-end-ASR-Pytorch/my_exp/dataB", type=str, help='Path to our raw dataB dataset')
parser.add_argument('--feature_type', default='fbank', type=str, help='Feature type ( mfcc / fbank )', required=False)
parser.add_argument('--feature_dim', default=40, type=int, help='Dimension of feature', required=False)
parser.add_argument('--apply_delta', default=True, type=boolean_string, help='Append Delta', required=False)
parser.add_argument('--apply_delta_delta', default=False, type=boolean_string, help='Append Delta Delta', required=False)
parser.add_argument('--apply_cmvn', default=True, type=boolean_string, help='Apply CMVN on feature', required=False)
parser.add_argument('--output_path', default='.', type=str, help='Path to store output', required=False)
parser.add_argument('--n_jobs', default=-1, type=int, help='Number of jobs used for feature extraction', required=False)
parser.add_argument('--target', default='subword', type=str, help='Learning target ( phoneme / char / subword / word )', required=False)
parser.add_argument('--n_tokens', default=5000, type=int, help='Vocabulary size of target', required=False)

parser.add_argument('--spec_aug', default=False, type=boolean_string, help='use specaugment', required=False)
parser.add_argument('--save_spec', default=False, type=boolean_string, help='save spectrogram picture', required=False)
paras = parser.parse_args()


def sep_seq(seq):
    """
    return a list of sentence
    e.g. "Ada{asd}qe{qs}a" -> [A,s,a,{asd},q,e,{qs},a]

    :param seq: input sentence
    :return: list of sentence
    """
    temp = []
    is_eng_word = False
    word_temp = ""
    for c in seq:
        word_temp = word_temp + c
        if c == "{" or c == "}" or c == "<" or c == ">":
            is_eng_word = not is_eng_word
            if not is_eng_word:
                temp.append(word_temp)
                word_temp = ""
        elif not is_eng_word:
            temp.append(word_temp)
            word_temp = ""
    return temp

def encode_label(label, encode_table):
    encoded_label = []
    encoded_label.append(0)
    for a in label:
        if a not in encode_table:
            encoded_label.append(encode_table['{<unk>}'])
        else:
            encoded_label.append(encode_table[a])
    encoded_label.append(1)

    return(encoded_label)


############################# change data set path ####################################
output_dir = "/media/ee303/My_Passport/Joint/new_exp/dataA/preprocess"
if not os.path.exists(output_dir):os.makedirs(output_dir)

classes_path = os.path.join(output_dir, 'label.json')
train_manifest = os.path.join(output_dir, 'train_manifest.csv')
test_manifest = os.path.join(output_dir, 'test_manifest.csv')

E2C_file = os.path.join(output_dir,'E2C.json')
C2E_file = os.path.join(output_dir,'C2E.json')

########################################################################################

# encoder dictionary
with open(classes_path,"r") as f:
    classes = json.load(f)
encode_table = {'<sos>':0,'<eos>':1}
for i in range(len(classes)):
    encode_table[classes[i]] = 2 + i
with open(C2E_file) as label_file:
    C2E = json.load(label_file)

count = 2 + len(classes)
for i in C2E.keys():
    while i not in encode_table:
        encode_table[i] = count
        count = count + 1


# train and test data
with open(train_manifest,"r") as f:
    train_path = f.readlines()
train_set = [a.replace('\n','').split(",") for a in train_path]

with open(test_manifest,"r") as f:
    test_path = f.readlines()
test_set = [a.replace('\n','').split(",") for a in test_path]

train_df = pd.DataFrame(train_set)
test_df = pd.DataFrame(test_set)

train_df.columns = ["audio_path","label_path"]
test_df.columns = ["audio_path","label_path"]

########################

bpe_dir = os.path.join(output_dir, 'bpe_E2C')
if not os.path.exists(bpe_dir):os.makedirs(bpe_dir)

with open(E2C_file) as label_file:
    E2C = json.load(label_file)

with open(os.path.join(bpe_dir, "txt4LM.txt"), "w") as f:
    train_syl_list = []
    for i in train_df["label_path"]:
        with open(i, "r") as lf:
            train_syl_label = lf.readlines()[0].strip()
        for j in E2C.keys():
            if j in train_syl_label:
                train_syl_label = train_syl_label.replace(j, E2C[j])
        train_syl_list.append(train_syl_label)
        f.write("".join(train_syl_label) + "\n")
    train_df["syl_label"] = train_syl_list

    test_syl_list = []
    for k in test_df["label_path"]:
        with open(k, "r") as lf:
            test_syl_label = lf.readlines()[0].strip()
        for l in E2C.keys():
            if l in test_syl_label:
                test_syl_label = test_syl_label.replace(l, E2C[l])
        test_syl_list.append(test_syl_label)
    test_df["syl_label"] = test_syl_list


# Train BPE
from subprocess import call
call(['spm_train',
      '--input=' + os.path.join(bpe_dir, 'txt4LM.txt'),
      '--model_prefix=' + os.path.join(bpe_dir, 'bpe'),
      '--vocab_size=' + str(count),
      '--character_coverage=1.0'
      ])

########################

train_path = os.path.join(output_dir, 'train')
if not os.path.exists(train_path):os.makedirs(train_path)
test_path = os.path.join(output_dir, 'test')
if not os.path.exists(test_path):os.makedirs(test_path)


for s in ["train","test"]:
    print(s)
    cur_path = os.path.join(output_dir,s)
    file_path = []
    length = []
    label = []
    for index, row in globals()[s + "_df"].iterrows():
        if s == 'train':
            feature = extract_feature(str(row["audio_path"]),feature=paras.feature_type,dim=paras.feature_dim,\
                cmvn=paras.apply_cmvn,delta=paras.apply_delta,delta_delta=paras.apply_delta_delta,spec_aug=paras.spec_aug,save_spec=paras.save_spec,save_feature=os.path.join(cur_path,row["audio_path"].split('/')[-1].replace('.wav','')))
        elif s =='test':
            feature = extract_feature(str(row["audio_path"]), feature=paras.feature_type, dim=paras.feature_dim, \
                                      cmvn=paras.apply_cmvn, delta=paras.apply_delta,delta_delta=paras.apply_delta_delta, spec_aug=False, save_spec=paras.save_spec, save_feature=os.path.join(cur_path,row["audio_path"].split('/')[-1].replace('.wav', '')))
        encoded = encode_label(sep_seq(row["syl_label"]),encode_table)
        encoded = [str(a) for a in encoded]
        length.append(feature)
        file_path.append(os.path.join(s,row["audio_path"].split('/')[-1].replace('.wav','.npy')))
        label.append("_".join(encoded))

    processed_df = pd.DataFrame(data={'file_path':file_path,'length':length,'label':label})
    processed_df = processed_df.sort_values(by=["length"], ascending=False).reset_index(drop=True)
    processed_df.to_csv(os.path.join(output_dir,s+'.csv'))

with open(os.path.join(output_dir,"mapping.pkl"), "wb") as fp:
    pickle.dump(encode_table, fp)


print('All done, saved at',output_dir,'exit.')