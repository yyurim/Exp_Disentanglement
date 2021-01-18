import json
import os, sys
import argparse
import numpy as np
import time
import torch
import torch.optim as optim
from loss import LogManager, calc_gaussprob, calc_kl_vae
import pickle
import model_exp as model
from data_manager import get_loader, make_spk_vector, make_one_hot_vector, get_all_target_idx
from itertools import combinations

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_sp(feat_dir, num_mcep=36):
    feat_path = os.path.join(feat_dir, 'feats.p')
    with open(feat_path, 'rb') as f:
        sp, _, _, _, _ = pickle.load(f)
    return sp

def calc_parm_num(model):

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def update_parm(opt_list, loss, epoch):
    
    for opt in opt_list:
        opt.zero_grad()
    loss.backward()
    for opt in opt_list:
        opt.step()


"""
VAE 1: Vanila
VAE 2: Decoder Speaker vector
VAE 3: All Speaker vector
MD: Multi Decoder

SI: Minimize speaker info (cross entropy) of latent
I: Minimize speaker entropy of latent

LI: Maximize ppg info of latent => ALC: ppg loss in converted x
AC: speaker loss in converted x

SC: l1(latent - cycle latent)
CC: cycle loss

GAN : discriminator
"""

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str)

parser.add_argument('--conf',type=str,default='')
parser.add_argument('--REC', type=float, default=1)
parser.add_argument('--KL', type=float, default=0.1)
parser.add_argument('--GAN', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--model_dir', default='')
parser.add_argument('--lr', type=float, default=1)

parser.add_argument('--epochs',type=int, default=2000)

args = parser.parse_args()
assert args.model_type in ["VAE1", "VAE2", "VAE3", "MD"]


if args.conf != "":
    with open(args.conf, 'r') as f:
        conf = json.load(f)
    args.REC = conf["REC"]
    args.KL = conf["KL"]

    print(args)




#initial seed 
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(args.seed)

# Data load
SPK_LIST = ['F1','M1','F2','M2']
TOTAL_SPK_NUM = len(SPK_LIST)

SP_DICT_TRAIN = {
    spk_id:load_sp(os.path.join("data_seoulNarrative","train", spk_id)) 
    for spk_id in SPK_LIST
}

SP_DICT_DEV = dict()
for spk_id in SPK_LIST:
    sps = []
    for _, _, file_list in os.walk(os.path.join("data_seoulNarrative", "dev", spk_id)):
        for file_id in file_list:
            utt_id = file_id.split(".")[0]
            if utt_id == "ppg36":
                continue
            file_path = os.path.join("data_seoulNarrative", "dev", spk_id, file_id)
            coded_sp, f0, ap = load_pickle(file_path)
            sps.append(coded_sp)
    SP_DICT_DEV[spk_id]=sps

# Model initilaization
model_dir = args.model_dir

######################################
#   do not hard coding
######################################
coef={"rec": args.REC, "adv": 0.0, "kl": args.KL}
print("coefficient !!",coef)
print(model_dir)
os.makedirs(model_dir+"/parm", exist_ok=True)

latent_dim=8

is_MD=True if args.model_type == "MD" else False

lr = 1

vae = model.VAE(style_dim=TOTAL_SPK_NUM, latent_dim=latent_dim, vae_type=args.model_type)
vae.cuda()
vae_opt = optim.Adam(vae.parameters(), lr=lr)
vae_sch = optim.lr_scheduler.LambdaLR(optimizer=vae_opt, lr_lambda=lambda epoch: -(1e-2/args.epochs)*epoch+1e-2)
print(calc_parm_num(vae))
print(vae)

# 8 16
# (0-499) (500-999)
epochs = args.epochs
print("Training Settings")
print("LR",lr)
print("Number of epochs",epochs)
print(".....................")
lm = LogManager()
lm.alloc_stat_type_list(["rec_loss", "kl_loss", "total_loss"])

total_time = 0
min_dev_loss = 9999999999999999
min_epoch = 0
d_epoch = 1

# print(Enc)
# print(Dec)

batch_size = 8
n_frames = 128



for epoch in range(epochs+1):
    print("EPOCH:   {}  LearningRate:   {}".format(epoch, vae_sch.get_last_lr()[0]))
    lm.init_stat()  

    start_time = time.time()
    # VAE Training
    vae.train()
    
    train_loader = get_loader(SP_DICT_TRAIN, batch_size, n_frames=n_frames, shuffle=True, is_MD=is_MD)

    for self_idx, self_coded_mcep in train_loader:
    
        one_hot_self = make_spk_vector(self_idx, TOTAL_SPK_NUM, batch_size, is_MD)

        self_z_mu, self_z_logvar, self_z, x_prime_mu, x_prime_logvar, x_prime = vae(x=self_coded_mcep,one_hot_src=one_hot_self, one_hot_tar=one_hot_self)

        rec_loss = -calc_gaussprob(self_coded_mcep, x_prime_mu, x_prime_logvar)
        kl_loss = calc_kl_vae(self_z_mu, self_z_logvar)      

        total_loss = coef["rec"] * rec_loss + coef["kl"] * kl_loss

        update_parm([vae_opt], total_loss, epoch)

        # print('\n\n')

        # write to log
        lm.add_torch_stat("rec_loss", rec_loss)
        lm.add_torch_stat("kl_loss", kl_loss)
        lm.add_torch_stat("total_loss", total_loss)

    print("Train:", end=' ')
    
    lm.print_stat()

    # VAE Evaluation
    lm.init_stat()
    vae.eval()
    # Enc.eval()
    # if is_MD:
    #     for dec in Dec_group.values():
    #         dec.eval()
    # else:
    #     Dec.eval()
    
    dev_loader = get_loader(SP_DICT_DEV, 1, n_frames=n_frames, shuffle=False, is_MD=is_MD)

    for self_idx, self_coded_mcep in dev_loader:
    
        one_hot_self = make_spk_vector(self_idx, TOTAL_SPK_NUM, batch_size, is_MD)
        
        with torch.no_grad():
            self_z_mu, self_z_logvar, self_z, x_prime_mu, x_prime_logvar, x_prime = vae(x=self_coded_mcep,one_hot_src=one_hot_self, one_hot_tar=one_hot_self)

            rec_loss = -calc_gaussprob(self_coded_mcep, x_prime_mu, x_prime_logvar)
            kl_loss = calc_kl_vae(self_z_mu, self_z_logvar)      

            total_loss = coef["rec"] * rec_loss + coef["kl"] * kl_loss
        
        lm.add_torch_stat("rec_loss", rec_loss)
        lm.add_torch_stat("kl_loss", kl_loss)
        lm.add_torch_stat("total_loss", total_loss)
    
    print("DEV:", end=' ')
    lm.print_stat()
    end_time = time.time()

    total_time += (end_time - start_time)

    print(".....................")

    vae_sch.step()
    
    cur_loss = lm.get_stat("total_loss")
    if np.isnan(cur_loss):
        print("Nan at",epoch)
        break


    if min_dev_loss > cur_loss:
        min_dev_loss = cur_loss
        min_epoch = epoch


    torch.save(vae.state_dict(), os.path.join(model_dir,"parm",str(epoch)+"_base.pt"))

    ####################################################
print("***********************************")
print("Model name:",model_dir.split("/")[-1])
print("TIME PER EPOCH:",total_time/(epochs+1))
print("Final Epoch:",min_epoch, min_dev_loss)
print("***********************************")
# min_epoch=epochs

os.system("cp "+os.path.join(model_dir,"parm",str(min_epoch)+"_base.pt")+" "+os.path.join(model_dir,"final_base.pt"))

print("\nsaved optimal model? >> ", os.path.isfile(os.path.join(model_dir,"final_base.pt")))

