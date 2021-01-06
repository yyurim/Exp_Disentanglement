import numpy as np
import torch

########################

def split_train_dev(datadict, train_percent=0.8):
    train_dict = dict()
    dev_dict = dict()
    for spk_id, cur_data in datadict.items():
        datanum = len(cur_data)
        train_num = int(datanum*train_percent)
        train_dict[spk_id]=cur_data[:train_num]
        dev_dict[spk_id]=cur_data[train_num:]
    return train_dict, dev_dict

def extract_target_from_ppg(ppg_mat, window=4):
    n_frames = ppg_mat.shape[-1]
    assert n_frames % window == 0, str(n_frames)+"\t"+str(window)
    target_list = []
    for start_idx in range(0, n_frames, window):
        end_idx = start_idx + window
        
        cur_ppg = np.sum(ppg_mat[:, start_idx:end_idx], axis=1)
        target_idx = np.argmax(cur_ppg)
        target_list.append(target_idx)
    
    return target_list

def sample_train_data(sp_list, n_frames=128, shuffle=False, ppg_list=None):
    """
    Input: [(D, T1), (D, T2), ... ]
    Output: [(D, 128), (D, 128), ... ]
    """

    total_num = len(sp_list)
    feat_idxs = np.arange(total_num)
    if shuffle:
        np.random.shuffle(feat_idxs)

    sp_mat_list = []
    target_list = []
    for idx in feat_idxs:
        cur_data = sp_list[idx]
        cur_data_frames = cur_data.shape[-1]
        
        assert cur_data_frames >= n_frames, "Too short SP"
        
        start_idx = np.random.randint(cur_data_frames - n_frames + 1)
        end_idx = start_idx + n_frames
        
        cur_sp_mat = cur_data[:, start_idx:end_idx]
        sp_mat_list.append(cur_sp_mat)
        if ppg_list is not None:
            cur_ppg = ppg_list[idx]
            cur_ppg_mat = cur_ppg[:, start_idx:end_idx]
            cur_target = extract_target_from_ppg(cur_ppg_mat, window=4)
            target_list.append(cur_target)

    result = np.array(sp_mat_list)
    targets = None if ppg_list == None else np.array(target_list)
    return result, targets

def feat_loader_MD(sp_dict, batch_size, n_frames=128, shuffle=False, ppg_dict=None):
    """
    spk_labs: int
    """
    total_feat = []
    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):
        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        spk_feats = []
        spk_targets = []
        sampled_sp, targets = sample_train_data(sp_list, n_frames=n_frames, ppg_list=ppg_list)
        
        # crawl wav sample
        for cur_sp in sampled_sp:
            spk_feats.append(cur_sp)
        
        if ppg_dict != None:
            for cur_target in targets:
                spk_targets.append(cur_target)

        # shuffle and pack        
        if shuffle:
            np.random.shuffle(spk_feats)
        feat_num = len(spk_feats)
        for start_idx in range(0, feat_num, batch_size):
            end_idx = start_idx + batch_size
            sps = spk_feats[start_idx:end_idx]
            labs = None if ppg_dict == None else spk_targets[start_idx:end_idx]
            
            total_feat.append((spk_idx, sps, labs))
    
    total_num = len(total_feat)
    if shuffle:
        np.random.shuffle(total_feat)
    
    for cur_idx in range(0, total_num):
        spk_idx, sps, labs = total_feat[cur_idx]

        x = np.expand_dims(sps, axis=1)
        x = torch.Tensor(x).float().cuda()

        if labs != None:
            t = torch.Tensor(labs).long().cuda()
            x = (x, t)

        yield x, spk_idx

def feat_loader_single(sp_dict, batch_size, n_frames=128, shuffle=False, ppg_dict=None):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    total_feat = []
    total_ppgs = []

    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):
        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        sampled_sp, targets = sample_train_data(sp_list, n_frames=n_frames, ppg_list=ppg_list)
        for cur_sp in sampled_sp:
            total_feat.append(
                (spk_idx, cur_sp)
            )
        if ppg_list is not None:
            for target in targets:
                total_ppgs.append(target)
    
    total_num = len(total_feat)
    total_idxs = np.arange(total_num)
    # print(total_idxs)
    # for src one hot
    src_idxs = [spk_idx for spk_idx, _ in total_feat]

    if shuffle:
        np.random.shuffle(total_idxs)
    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        index_list = total_idxs[start_idx:end_idx]

        # for src one hot
        src_list = src_idxs[start_idx:end_idx]

        x=[]
        t=[]
        spk_idxs = []

        for cur_idx in index_list:
            spk_idx, sp = total_feat[cur_idx]
            spk_idxs.append(spk_idx)
            x.append(sp)
            
            if ppg_dict is not None:
                t.append(total_ppgs[cur_idx])

        # post processing
        x = np.expand_dims(x, axis=1)
        x = torch.Tensor(x).float().cuda()

        if ppg_dict is not None:
            t = torch.Tensor(t).long().cuda()
            x = (x, t)
        
        yield src_list, x, spk_idxs

def get_loader(SP_DICT, batch_size, n_frames=128, shuffle=False, PPG_DICT=None, is_MD=False):
    data_loader = None
    if is_MD:
        data_loader = feat_loader_MD(SP_DICT, batch_size, n_frames=n_frames, shuffle=shuffle, ppg_dict=PPG_DICT)
    else:
        data_loader = feat_loader_single(SP_DICT, batch_size, n_frames=n_frames, shuffle=shuffle, ppg_dict=PPG_DICT)

    return data_loader

########################
"""
MD: spk_idx => A_y (batch_len, vec_dim)
VAE: spk_idxs => A_y 

spk_idx => make_spk_vector, make_spk_target
"""
########################
def make_one_hot_vector(spk_idx, spk_num):
    vec = np.zeros(spk_num)
    vec[spk_idx] = 1.0
    return vec

def expand_spk_vec(spk_vec, batch_len):
    spk_vec = np.expand_dims(spk_vec, axis=0)
    y = np.repeat(spk_vec, batch_len, axis=0)
    y = torch.Tensor(y).float().cuda()
    return y

def make_spk_vector(spk_idxs, spk_num, batch_len=0, is_MD=False):
    A_y = []
    if is_MD:
        spk_idx = spk_idxs
        spk_vec = make_one_hot_vector(spk_idx, spk_num)
        A_y = expand_spk_vec(spk_vec, batch_len)
    else:
        for spk_idx in spk_idxs:
            spk_vec = make_one_hot_vector(spk_idx, spk_num)
            A_y.append(spk_vec)
        A_y = torch.Tensor(A_y).float().cuda()
    return A_y

########################
def make_lab(spk_idx, batch_len):
    t = torch.Tensor([spk_idx]).long().cuda()
    t = t.repeat((batch_len))
    return t

def make_spk_target(spk_idxs, batch_len=0, is_MD=False):
    A_spk_lab = []
    if is_MD:
        spk_idx = spk_idxs
        A_spk_lab = make_lab(spk_idx, batch_len)
    else:
        A_spk_lab = [spk_idx for spk_idx in spk_idxs]
        A_spk_lab = torch.Tensor(A_spk_lab).long().cuda()

    return A_spk_lab

########################

def get_all_target_idx(A_spk_idxs, spk_num, is_MD=False):
    """
    A_spk_idxs: [int, int, int, ...] or int
    B_spk_idxs: [[int, int, int], [int, int, int], ...] or [int, int, int]
    """
    result=[]

    if is_MD:
        B_spk_idx_list = []
        src_spk_idx = A_spk_idxs
        for trg_spk_idx in range(spk_num):
            # skip when source speaker is same with target speaker 
            if src_spk_idx == trg_spk_idx:
                continue
            B_spk_idx_list.append(trg_spk_idx)
        result = np.array(B_spk_idx_list)
    else:
        for src_spk_idx in A_spk_idxs:
            B_spk_idx_list = []
            for trg_spk_idx in range(spk_num):
                # skip when source speaker is same with target speaker 
                if src_spk_idx == trg_spk_idx:
                    continue
                B_spk_idx_list.append(trg_spk_idx) # (3)
            result.append(B_spk_idx_list) # (batch_len, 3)
        result = np.swapaxes(np.array(result), 0, 1) # (3, batch_len)

    return result