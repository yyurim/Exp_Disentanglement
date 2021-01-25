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
        # start_idx = np.random.randint(cur_data_frames - n_frames)
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


def multiple_sample_train_data(sp_list, n_frames=128, shuffle=False, ppg_list=None):
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

        cur_sp_mat = []
        cur_target = []
        for seg_idx in range(0,cur_data_frames,n_frames):
            start_idx = seg_idx
            end_idx = seg_idx+n_frames

            if end_idx >= cur_data_frames:
                break

            cur_sp_mat.append(cur_data[:, start_idx:end_idx])

            if ppg_list is not None:
                cur_ppg = ppg_list[idx]
                cur_ppg_mat = cur_ppg[:, start_idx:end_idx]
                cur_target.append(extract_target_from_ppg(cur_ppg_mat, window=4))
                

        sp_mat_list.extend(cur_sp_mat)
        if ppg_list is not None:
            target_list.extend(cur_target)

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

def feat_loader_single(sp_dict, batch_size, n_frames=128, shuffle=False, ppg_dict=None, is_AC=False):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    TOTAL_SPK_NUM = len(sp_dict)
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

    if shuffle:
        np.random.shuffle(total_idxs)

    if is_AC:
        np.random.shuffle(total_idxs)
    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        index_list = total_idxs[start_idx:end_idx]

        spk_idxs = []

        x=[]
        t=[]

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
        
        yield spk_idxs, x


def feat_loader_multiple(sp_dict, batch_size, n_frames=128, shuffle=False, ppg_dict=None, is_AC=False):
    TOTAL_SPK_NUM = len(sp_dict)
    total_feat = []
    total_ppgs = []

    for spk_idx, (spk_id, sp_list) in enumerate(sp_dict.items()):
        ppg_list = None if ppg_dict == None else ppg_dict[spk_id]
        sampled_sp, targets = multiple_sample_train_data(sp_list, n_frames=n_frames, ppg_list=ppg_list)

        spk_sp_idxs_num = len(sampled_sp)
        spk_sp_idxs = np.arange(spk_sp_idxs_num)

        np.random.shuffle(spk_sp_idxs)

        # minimum 291
        spk_sp_idxs = spk_sp_idxs[:200]
        for i in spk_sp_idxs:
            total_feat.append(
                (spk_idx, sampled_sp[i])
            )

            if ppg_list is not None:
                total_ppgs.append(targets[i])
    
    
    total_num = len(total_feat)
    sp_per_spk = total_num//TOTAL_SPK_NUM

    total_idxs = np.arange(total_num)
    total_idxs_shuffled = np.arange(total_num)

    spk_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        s = spk * sp_per_spk
        e = s + sp_per_spk
        np.random.shuffle(total_idxs_shuffled[s:e])
        spk_sp_idx.append(total_idxs_shuffled[s:e])


    tar_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        for i in range(sp_per_spk):
            tar_sp_idx_pt = []
            for tar in range(TOTAL_SPK_NUM):
                if tar == spk:
                    continue
                tar_sp_idx_pt.append(spk_sp_idx[tar][i])

            tar_sp_idx.append(tar_sp_idx_pt)


    total_pairs = []
    for i in range(total_num):
        src = total_idxs[i]
        tars = tar_sp_idx[i]
        total_pairs.append([src,tars])
    
    if shuffle:
        np.random.shuffle(total_pairs)

    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        pairs = total_pairs[start_idx:end_idx]

        # for src one hot
        src_idxs = []

        src_x=[]
        src_t=[]

        tar_idxs = []

        tar_x=[]
        tar_t=[]

        for src, targets in pairs:

            src_idx, src_sp = total_feat[src]

            src_idxs.append(src_idx)
            src_x.append(src_sp)

            if ppg_dict is not None:
                src_t.append(total_ppgs[src])

            tar_idxs_pt = []

            tar_x_pt = []
            tar_t_pt = []

            for tar in targets:
                tar_idx, tar_sp = total_feat[tar]

                tar_idxs_pt.append(tar_idx)
                tar_x_pt.append(tar_sp)
        
                if ppg_dict is not None:
                    tar_t_pt.append(total_ppgs[tar])
            
            tar_idxs.append(tar_idxs_pt)

            tar_x.append(tar_x_pt)
            tar_t.append(tar_t_pt)

        # post processing
        src_x = np.expand_dims(src_x, axis=1)
        src_x = torch.Tensor(src_x).float().cuda()
        # print(src_x.shape)


        tar_idxs = list(np.array(tar_idxs).swapaxes(0,1))
        tar_x = np.expand_dims(tar_x, axis=1)
        tar_x = torch.Tensor(tar_x).float().cuda()
        tar_x = tar_x.permute(2,0,1,3,4)
        # print(tar_x.shape)

        if ppg_dict is not None:
            src_t = torch.Tensor(src_t).long().cuda()
            src_x = (src_x, src_t)

            # tar_t = torch.Tensor(tar_t).long().cuda()
            # tar_t = tar_t.permute(2,0,1,3,4)

            # tar_x = (tar_x, tar_t)
        
        yield src_idxs, src_x, tar_idxs, tar_x



def feat_loader_pair(sp_dict, batch_size, n_frames=128, shuffle=False, ppg_dict=None,is_MD=False):
    """
    spk_labs: list of ints [int, int, int, ...]
    """
    TOTAL_SPK_NUM = len(sp_dict)
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
    sp_per_spk = total_num//TOTAL_SPK_NUM

    total_idxs = np.arange(total_num)
    total_idxs_shuffled = np.arange(total_num)

    spk_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        s = spk * sp_per_spk
        e = s + sp_per_spk
        np.random.shuffle(total_idxs_shuffled[s:e])
        spk_sp_idx.append(total_idxs_shuffled[s:e])


    tar_sp_idx = []
    for spk in range(TOTAL_SPK_NUM):
        for i in range(sp_per_spk):
            tar_sp_idx_pt = []
            for tar in range(TOTAL_SPK_NUM):
                if tar == spk:
                    continue
                tar_sp_idx_pt.append(spk_sp_idx[tar][i])

            tar_sp_idx.append(tar_sp_idx_pt)


    total_pairs = []
    for i in range(total_num):
        src = total_idxs[i]
        tars = tar_sp_idx[i]
        total_pairs.append([src,tars])
    
    if shuffle:
        np.random.shuffle(total_pairs)

    
    for start_idx in range(0, total_num, batch_size):
        end_idx = start_idx + batch_size
        pairs = total_pairs[start_idx:end_idx]

        # for src one hot
        src_idxs = []

        src_x=[]
        src_t=[]

        tar_idxs = []

        tar_x=[]
        tar_t=[]

        for src, targets in pairs:

            src_idx, src_sp = total_feat[src]

            src_idxs.append(src_idx)
            src_x.append(src_sp)

            if ppg_dict is not None:
                src_t.append(total_ppgs[src])

            tar_idxs_pt = []

            tar_x_pt = []
            tar_t_pt = []

            for tar in targets:
                tar_idx, tar_sp = total_feat[tar]

                tar_idxs_pt.append(tar_idx)
                tar_x_pt.append(tar_sp)
        
                if ppg_dict is not None:
                    tar_t_pt.append(total_ppgs[tar])
            
            tar_idxs.append(tar_idxs_pt)

            tar_x.append(tar_x_pt)
            tar_t.append(tar_t_pt)

        # post processing
        src_x = np.expand_dims(src_x, axis=1)
        src_x = torch.Tensor(src_x).float().cuda()


        tar_idxs = list(np.array(tar_idxs).swapaxes(0,1))
        tar_x = np.expand_dims(tar_x, axis=1)
        tar_x = torch.Tensor(tar_x).float().cuda()
        tar_x = tar_x.permute(2,0,1,3,4)

        if ppg_dict is not None:
            src_t = torch.Tensor(src_t).long().cuda()
            src_x = (src_x, src_t)

            # tar_t = torch.Tensor(tar_t).long().cuda()
            # tar_t = tar_t.permute(2,0,1,3,4)

            # tar_x = (tar_x, tar_t)
        
        yield src_idxs, src_x, tar_idxs, tar_x
# 

def get_loader(SP_DICT, batch_size, n_frames=128, shuffle=False, PPG_DICT=None, is_MD=False, is_AC=False):
    data_loader = None
    if is_MD:
        data_loader = feat_loader_MD(SP_DICT, batch_size, n_frames=n_frames, shuffle=shuffle, ppg_dict=PPG_DICT)
    else:
        data_loader = feat_loader_single(SP_DICT, batch_size, n_frames=n_frames, shuffle=shuffle, ppg_dict=PPG_DICT, is_AC=is_AC)

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