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
        sp, _, _, _, _ = pickle.load(f)
    return sp

def load_ppg(feat_dir, num_mcep=36):
    ppg_path = os.path.join(feat_dir, 'ppg{}.p'.format(num_mcep))
    with open(ppg_path, 'rb') as f:
        ppg = pickle.load(f)
    return ppg

def calc_parm_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str) # VAE3 MD

parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--model_dir', default='pretrainSI')
parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--c_lr', type=float, default=2.5*1e-5)

parser.add_argument('--lr_sch',type=str, default='linear15')
parser.add_argument('--epochs',type=int, default=1000)

parser.add_argument('--baseline',type=str, default='')
parser.add_argument('--disentanglement', type=str, default='')
parser.add_argument('--ws', type=int, default=1)
parser.add_argument('--spk', type=str, default='')

args = parser.parse_args()
assert args.model_type in ["VAE1", "VAE2", "VAE3", "MD"]

is_MD=True if args.model_type=="MD" else False

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(args.seed)


# Data load
if args.spk != "":
    SPK_LIST = ['VCC2SF1','VCC2SF2','VCC2SM1','VCC2SM2','VCC2SF3','VCC2SF4','VCC2SM3','VCC2SM4'] 
else:
    SPK_LIST = ['VCC2SF1','VCC2SF2','VCC2SM1','VCC2SM2'] 
# SPK_LIST = ['VCC2SF1','VCC2SF2','VCC2SM1','VCC2SM2'] 
# SPK_LIST = ['F1','M1','F2','M2']
TOTAL_SPK_NUM = len(SPK_LIST)

PPG_DICT_TRAIN = {
    spk_id:load_ppg(os.path.join("data","train", spk_id)) 
    for spk_id in SPK_LIST
}

PPG_DICT_DEV = {
    spk_id:load_ppg(os.path.join("data","dev", spk_id)) 
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
            _, coded_sp, f0, ap = load_pickle(file_path)
            sps.append(coded_sp)
    SP_DICT_DEV[spk_id]=sps

# Model initilaization
model_dir = args.model_dir
os.makedirs(model_dir,exist_ok=True)

latent_dim=8

lr = 1
c_lr = args.c_lr

batch_size = 8

epochs = args.epochs

total_time = 0

min_dev_loss = 9999999999999999
min_epoch = 0
d_epoch = 1

pre_vae = model.VAE(style_dim=TOTAL_SPK_NUM, latent_dim=latent_dim, vae_type=args.model_type, weight_sharing=args.ws )
pre_vae.load_state_dict(torch.load(args.baseline))
pre_vae.cuda()
pre_vae.eval()

spk_C = model.LatentClassifier(latent_dim=latent_dim, label_num=TOTAL_SPK_NUM)
spk_C.cuda()
spk_C_opt = optim.Adam(spk_C.parameters(), lr=c_lr)
# spk_C_sch = optim.lr_scheduler.LambdaLR(optimizer=spk_C_opt, lr_lambda=lambda epoch: c_lr*(-(1e-2/(epochs+1))*epoch+1e-2))
print(calc_parm_num(spk_C))
print(spk_C)

torch.save(spk_C.state_dict(), os.path.join(model_dir,"si_{}.pt".format(epochs)))

lm = LogManager()
lm.alloc_stat_type_list(["train_loss", "train_acc", "dev_loss", "dev_acc"])

for epoch in range(epochs+1):
    print("SI Epoch: {}     LearningRate:   {}".format(epoch, spk_C_opt.param_groups[0]['lr']))

    lm.init_stat()  

    spk_C.train()
    train_loader = dm.feat_loader_single(SP_DICT_TRAIN, batch_size, shuffle=True)      
    for self_idx, coded_mcep in train_loader:
        
        one_hot_self = dm.make_spk_vector(self_idx, TOTAL_SPK_NUM, batch_size, is_MD)
        
        total_loss = 0.0
        z_mu, z_logvar, z, x_prime_mu, x_prime_logvar, x_prime = pre_vae(x=coded_mcep,one_hot_src=one_hot_self, one_hot_tar=one_hot_self)
        
        # Latent Classifier
        self_vec = dm.make_spk_target(self_idx, batch_size, is_MD=False)
        predicted_self = spk_C(z)
        si_loss = nllloss(predicted_self, self_vec)
        si_err = calc_err(predicted_self, self_vec)

        spk_C_opt.zero_grad()
        si_loss.backward()
        spk_C_opt.step()

        lm.add_torch_stat("train_loss", si_loss)
        lm.add_torch_stat("train_acc", 1-si_err)

    print("Train:", end=' ')
    lm.print_stat()

    lm.init_stat()
    spk_C.eval()
    dev_loader = dm.feat_loader_single(SP_DICT_DEV, batch_size, shuffle=True)      
    for self_idx, coded_mcep in dev_loader:
        
        one_hot_self = dm.make_spk_vector(self_idx, TOTAL_SPK_NUM, batch_size, is_MD)
        
        total_loss = 0.0
        z_mu, z_logvar, z, x_prime_mu, x_prime_logvar, x_prime = pre_vae(x=coded_mcep,one_hot_src=one_hot_self, one_hot_tar=one_hot_self)
        
        # Latent Classifier
        self_vec = dm.make_spk_target(self_idx, batch_size, is_MD=False)
        predicted_self = spk_C(z)
        si_loss = nllloss(predicted_self, self_vec)
        si_err = calc_err(predicted_self, self_vec)

        lm.add_torch_stat("dev_loss", si_loss)
        lm.add_torch_stat("dev_acc", 1-si_err)
    
    print("DEV:", end=' ')
    lm.print_stat()
    print(".....................")
    # spk_C_sch.step()

torch.save(spk_C.state_dict(), os.path.join(model_dir,"si_{}.pt".format(epochs)))
