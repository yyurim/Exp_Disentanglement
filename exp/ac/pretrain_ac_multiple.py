import os, sys
import argparse
import numpy as np
import time
import torch
import torch.optim as optim
from loss import LogManager, calc_gaussprob, calc_kl_vae, nllloss, calc_entropy, calc_err, l1loss, calc_entropy_log
import pickle
import model
from itertools import combinations
import data_manager as dm
import json

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_sp(feat_dir, num_mcep=36):
    feat_path = os.path.join(feat_dir, 'feats.p')
    with open(feat_path, 'rb') as f:
        sp, sp_m, sp_s, logf0_m, logf0_s = pickle.load(f)
    return sp

def load_ppg(feat_dir, num_mcep=36):
    ppg_path = os.path.join(feat_dir, 'ppg{}.p'.format(num_mcep))
    with open(ppg_path, 'rb') as f:
        ppg = pickle.load(f)
    return ppg

def calc_parm_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def update_parm(opt_list, loss):
    for opt in opt_list:
        opt.zero_grad()
    loss.backward()
    for opt in opt_list:
        opt.step()

def set_DEC(DEC, mode, is_MD=False):
    assert mode in ['train', 'eval']
    if is_MD:
        for dec in DEC.values():
            if mode=='train':
                dec.train()
            if mode=="eval":
                dec.eval()
    else:
        if mode=='train':
            DEC.train()
        if mode=="eval":
            DEC.eval()
    

"""
VAE 1: Vanila
VAE 2: Decoder Speaker vector
VAE 3: All Speaker vector (S2S)
MD: Multi Decoder (S2S)

============ A2A ============

SI: Minimize speaker info (cross entropy) of latent 
I: Maximize latent entropy
LI: Maximize ppg info of latent 

============ A2B ============

AC: speaker loss in converted x
SC: l1(latent - cycle latent)
CC: cycle loss

GAN : discriminator
"""

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model_dir', default='')
parser.add_argument('--lr', type=float, default=0)
parser.add_argument('--c_lr', type=float, default=2.5*1e-5)

parser.add_argument('--epochs',type=int, default=2000)

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(args.seed)


# Data load
# SPK_LIST = ['F1','M1','F2','M2']
SPK_LIST = ['VCC2SF1','VCC2SF2','VCC2SM1','VCC2SM2'] 
TOTAL_SPK_NUM = len(SPK_LIST)

PPG_DICT_TRAIN = {
    spk_id:load_ppg(os.path.join("data","train", spk_id)) 
    for spk_id in SPK_LIST
}

SP_DICT_TRAIN = {
    spk_id:load_sp(os.path.join("data","train", spk_id)) 
    for spk_id in SPK_LIST
}

SP_DICT_DEV = dict()
for spk_id in SPK_LIST:
    sps = []
    for _, _, file_list in os.walk(os.path.join("data", "dev", spk_id)):
        for file_id in file_list:
            utt_id = file_id.split(".")[0]
            if utt_id == "ppg36":
                continue
            file_path = os.path.join("data", "dev", spk_id, file_id)
            _,coded_sp, f0, ap = load_pickle(file_path)
            sps.append(coded_sp)
    SP_DICT_DEV[spk_id]=sps
# Model initilaization
model_dir = args.model_dir

print(model_dir)
os.makedirs(model_dir, exist_ok=True)

latent_dim=8


# lr = 1e-3
c_lr = 1e-5*2.5

AC = model.DataClassifier(latent_dim=latent_dim, label_num=TOTAL_SPK_NUM)
AC.cuda()
AC_opt = optim.Adam(AC.parameters(), lr=c_lr)
AC_sch = optim.lr_scheduler.ExponentialLR(AC_opt, 0.5)

# 8 16
# (0-499) (500-999)
total_time = 0

min_dev_loss = 9999999999999999
min_epoch = 0
d_epoch = 1

lm = LogManager()
lm.alloc_stat_type_list(["train_loss", "train_acc", "dev_loss", "dev_acc"])
pretrain_epochs = args.epochs
batch_size = 8
print("Train AC")
for epoch in range(pretrain_epochs):
    print("AC EPOCH:   {}  LearningRate:   {}".format(epoch, AC_sch.get_last_lr()[0]))
    lm.init_stat()  
    # Train
    AC.train()
    train_loader = dm.feat_loader_multiple(SP_DICT_TRAIN, batch_size, shuffle=True, ppg_dict=PPG_DICT_TRAIN)
    for self_idx, (coded_mcep, _), _, _ in train_loader:

        x = dm.make_spk_target(self_idx, batch_size, is_MD=False)

        pred_x = AC(coded_mcep)
        spk_loss = nllloss(pred_x, x)
        spk_err = calc_err(pred_x, x)

        AC_opt.zero_grad()
        spk_loss.backward()
        AC_opt.step()

        lm.add_torch_stat("train_loss", spk_loss)
        lm.add_torch_stat("train_acc", 1.0 - spk_err)

    print("Train:", end=' ')
    lm.print_stat()
    lm.init_stat()
    # Dev
    AC.eval()
    dev_loader = dm.feat_loader_single(SP_DICT_DEV, batch_size, shuffle=False)
    for self_idx, coded_sp in dev_loader:

        x = dm.make_spk_target(self_idx, batch_size, is_MD=False)

        pred_x = AC(coded_sp)
        spk_loss = nllloss(pred_x, x)
        spk_err = calc_err(pred_x, x)

        lm.add_torch_stat("dev_loss", spk_loss)
        lm.add_torch_stat("dev_acc", 1.0 - spk_err)
    
    print("DEV:", end=' ')
    lm.print_stat()
    print(".....................")
    # AC_sch.step()
    AC.eval()

torch.save(AC.state_dict(), os.path.join(model_dir,"ac_{}.pt".format(pretrain_epochs)))


