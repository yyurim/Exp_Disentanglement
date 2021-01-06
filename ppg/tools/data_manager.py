import torch
from torch.utils.data import Dataset
import numpy as np
from random import shuffle

# data.expand_mat(feat_mat, left=5, right=5)
def expand_mat(feat_mat, left, right, memory_save=False, allow_zero=True):
    feat_dim = len(feat_mat[0])
    if allow_zero:
        vertical_mat = np.vstack(
                    [np.zeros((left, feat_dim),dtype=np.float32),
                    np.array(feat_mat, dtype=np.float32), 
                    np.zeros((right, feat_dim),dtype=np.float32)])
    else:
        vertical_mat=np.array(feat_mat, dtype=np.float32)
    if memory_save:
        return vertical_mat
    else:
        input_dim = feat_dim * (left+right+1)
        result_mat = []
        for feat_idx in range(len(vertical_mat)-left-right):
            result = vertical_mat[feat_idx:feat_idx+left+right+1]
            result = np.reshape(result, input_dim)
            result_mat.append(result)
        result_mat = np.vstack(result_mat)
        return result_mat

class Trainset(Dataset):
    def __init__(self, *args, **kwargs):
        super(Trainset, self).__init__()
        datadict = kwargs.get("datadict", dict())
        self.dataset=dict()
        self.left = kwargs.get("left", 0)
        self.right = kwargs.get("right", 0)
        self.memory_save = kwargs.get("memory_save", False)
        assert (len(datadict)>0), "Invalid dataset"

        self.utt_list = []
        for utt_id, feat_mat in datadict.items():
            sample_feat=feat_mat
            total_len = len(feat_mat)
            for feat_idx in range(total_len-self.left-self.right):
                self.utt_list.append((utt_id, feat_idx))
            result_mat = expand_mat(feat_mat, self.left, self.right, self.memory_save, allow_zero=True)
            self.dataset[utt_id] = result_mat     
        self.input_dim = len(sample_feat[0]) * (self.left + 1 + self.right)       

    def __getitem__(self, idx):
        utt_id, feat_idx = self.utt_list[idx]
        if self.memory_save:
            feat_frame = self.dataset[utt_id][feat_idx:feat_idx+self.left+self.right+1]
            feat_frame = np.reshape(feat_frame, self.input_dim)
        else:
            feat_frame = self.dataset[utt_id][feat_idx]
        return (utt_id, feat_idx, feat_frame)

    def __len__(self):
        return len(self.utt_list)

class Labelset:
    def __init__(self, *args, **kwargs):
        self.labeldict = kwargs.get("labeldict", dict())
        self.left = kwargs.get("left", 0)
        self.right = kwargs.get("right", 0)
    
    def get(self, utt_ids, feat_idxs):
        labels = [] 
        for utt_id, feat_idx in zip(utt_ids, feat_idxs):
            real_idx = feat_idx
            label = self.labeldict[utt_id][real_idx]
            labels.append(label)
        labels = np.array(labels)
        labels = torch.Tensor(labels).long()
        return labels

def make_pbw_lexicon(feat_path):
    # lexicon: {keyword: [phone1, phone2, ...]}
    lexicon = dict()
    with open(feat_path+"/text", 'r') as f:
        for line in f:
            word = line.split(" ")[2]
            phone_set = word.split("_")
            lexicon[word]=phone_set
    return lexicon

def make_pbw_keyword_map(feat_path):
    # keyword_map: {keyword: [utt_id1, utt_id2, ...]}
    keyword_dict = dict()
    with open(feat_path+"/text", 'r') as f:
        for line in f:
            utt_id = line.split(" ")[0]
            word = line.split(" ")[2]
        
            utt_list = keyword_dict.get(word,[])
            utt_list.append(utt_id)
            keyword_dict[word]=utt_list
    return keyword_dict

def make_timit_lexicon(lexicon_path):
    # lexicon: {keyword: [phone1, phone2, ...]}
    lexicon = dict()
    with open(lexicon_path, 'r') as f:
        for line in f:
            word = line.split(" ")[0]
            phone_set = line[:-1].split(" ")[1:]
            lexicon[word]=phone_set
    return lexicon

def make_timit_keyword_map(feat_path):
    # keyword_map: {keyword: [utt_id1, utt_id2, ...]}
    keyword_dict = dict()
    with open(feat_path+"/utt2spk", 'r') as f:
        for line in f:
            utt_id = line.split(" ")[0]
            word = utt_id.split("_")[1]
        
            utt_list = keyword_dict.get(word,[])
            utt_list.append(utt_id)
            keyword_dict[word]=utt_list
    return keyword_dict

def make_cnn_dataset(utt_dict, frame_size, step_size, verbose=False):
    segment_set = []
    for utt_id, frame_mat in utt_dict.items():
        frame_len = len(frame_mat)
        if frame_len < frame_size:
            if verbose:
                print("Skip "+utt_id)
            continue
        
        for start_idx in range(0, frame_len-frame_size+1, step_size):
            segment = frame_mat[start_idx:start_idx+frame_size]
            segment_set.append(segment)
        segment = frame_mat[frame_len-frame_size:]
        segment_set.append(segment)
    return segment_set

def get_total_cmvn_stats(feat_dict):
    total_x = []
    for feat_mat in feat_dict.values():
        total_x.extend(feat_mat)
    mean = np.mean(total_x, axis=0)
    var = np.var(total_x, axis=0)
    return mean, var

def apply_cmvn_total(feat_dict, mean, var=None):
    new_dict = dict()
    for utt_id, feat_mat in feat_dict.items():
        new_mat = feat_mat - mean
        if var is not None:
            new_mat /= var
        new_dict[utt_id] = new_mat
    return new_dict

def apply_cmvn_and_get_stats(feat_dict, apply_var=False):
    cmvn_stats = dict()
    new_dict = dict()
    for utt_id, feat_mat in feat_dict.items():
        ms = np.mean(feat_mat, axis=0)
        stat=[ms]
        new_mat = feat_mat - ms
        
        if apply_var:
            vs = np.var(feat_mat, axis=0)
            new_mat = new_mat / vs
            stat.append(vs)
        cmvn_stats[utt_id] = stat
        new_dict[utt_id] = new_mat
    return new_dict, cmvn_stats

def reverse_cmvn(feat_dict, stat_dict):
    new_dict = dict()
    for utt_id, feat_mat in feat_dict.items():
        stat = stat_dict[utt_id]
        if len(stat) == 2:
            new_mat = stat[1] * feat_mat + stat[0]
        else:
            new_mat = feat_mat + stat[0]
        new_dict[utt_id] = new_mat
    return new_dict

def make_cnn_spk_vec_dataset(utt_dict, spk_list, vec_dict, frame_size, step_size, verbose=False):
    spk_dict = {spk_id: idx for idx, spk_id in enumerate(spk_list)}

    segment_set = []
    for utt_id, frame_mat in utt_dict.items():
        frame_len = len(frame_mat)
        spk_id = utt_id.split("-")[0]
        spk_idx = spk_dict[spk_id]
        vec = vec_dict[spk_id]
        if frame_len < frame_size:
            if verbose:
                print("Skip "+utt_id)
            continue
        
        for start_idx in range(0, frame_len-frame_size+1, step_size):
            segment = frame_mat[start_idx:start_idx+frame_size]
            segment_set.append((segment, spk_idx, vec))
        segment = frame_mat[frame_len-frame_size:]
        segment_set.append((segment, spk_idx, vec))
    return segment_set

def fix_feat(feat_dict, ali_dict):
    real_utts = list(ali_dict.keys())
    result_dict = dict()
    for utt_id, feat_mat in feat_dict.items():
        if utt_id in real_utts:
            result_dict[utt_id]=feat_mat
    return result_dict

#######################################################
# Input: feat_dictionary, lab_dictionary
# Output: [feat*batch], lab-concat
import random
from math import ceil

def rnn_generator(feat_dict, ali_dict, batch_size, shuffle=True):
    utt_ids = list(feat_dict.keys())
    total_utts = len(utt_ids)
    iter_count = ceil(total_utts // batch_size)
    
    while True:
        if shuffle:
            random.shuffle(utt_ids)
        if batch_size == 1:
            for utt_id, feat_mat in feat_dict.items():
                cur_feat = [torch.Tensor(feat_mat).float().cuda()]
                cur_lab = torch.Tensor(ali_dict[utt_id]).long().cuda()
                yield cur_feat, cur_lab
        else:
            for iter_idx in range(iter_count):
                start_idx = iter_idx*batch_size
                end_idx = min((iter_idx+1)*batch_size, total_utts)

                cur_utts = utt_ids[start_idx:end_idx]
                cur_feats = []
                cur_labs = []

                for utt_id in cur_utts:
                    cur_feat = torch.Tensor(feat_dict[utt_id]).float().cuda()
                    cur_feats.append(cur_feat)

                    cur_lab = torch.Tensor(ali_dict[utt_id]).long().cuda()
                    cur_labs.append(cur_lab)
                cur_labs = torch.cat(cur_labs, dim=0)
                yield cur_feats, cur_labs   
        break
#######################################################

def crop_datadict(datadict, frame_size=128):
    # datadict = {utt_id: feat_mat}
    result_dict=dict()
    for utt_id, feat_mat in datadict.items():
        total_len = len(feat_mat)
        if total_len < frame_size:
            continue
        result_dict[utt_id] = feat_mat
    return result_dict

def sample_datadict(datadict, frame_size=128, batch_size=8):
    # datadict = {utt_id: feat_mat}
    uttlist = list(datadict.keys())
    for start_utt in range(0, len(uttlist), batch_size):
        end_utt = start_utt+batch_size
        cur_utts = uttlist[start_utt:end_utt]
        cur_batch = []
        for utt_id in cur_utts:
            feat_mat = datadict[utt_id]
            total_len = len(feat_mat)
            
            sidx = np.random.randint(total_len - frame_size + 1)
            eidx = sidx+frame_size
            cur_segment = feat_mat[sidx:eidx]
            cur_batch.append(cur_segment)
        yield cur_batch

def global_cmvn(data_dict, feat_dim=13):
    total_sum = np.zeros(feat_dim)
    total_square_sum = np.zeros(feat_dim)
    total_count = 0
    for utt_id, feat_mat in data_dict.items():
        total_sum += np.sum(feat_mat, axis=0)
        total_square_sum += np.sum(pow(feat_mat,2), axis=0)
        total_count += len(feat_mat)
    
    mean = total_sum/total_count
    var = total_square_sum/total_count - pow(mean, 2)
    std = np.sqrt(var)

    for utt_id, feat_mat in data_dict.items():
        data_dict[utt_id] = (feat_mat - mean) / std

    return data_dict