import os, sys
import model
import argparse
import soundfile
import torch
import numpy as np
import pickle
import json

import torch.optim as optim
from loss import LogManager, calc_gaussprob, calc_kl_vae
from data_manager import get_loader, make_spk_vector, make_one_hot_vector
from itertools import combinations

from speech_tools import world_decode_mc, world_speech_synthesis

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


SPK_LIST = ['F1','M1','F2','M2']
TOTAL_SPK_NUM = len(SPK_LIST)

SP_DICT_TEST = dict()
for spk_id in SPK_LIST:
    sps = []
    for _, _, file_list in os.walk(os.path.join("data_seoulNarrative", "test", spk_id)):
        for file_id in file_list:
            utt_id = file_id.split(".")[0]
            if utt_id == "ppg36":
                continue
            file_path = os.path.join("data_seoulNarrative", "test", spk_id, file_id)
            _, coded_sp, f0, ap = load_pickle(file_path)
            sps.append(coded_sp)
    SP_DICT_TEST[spk_id]=sps

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='')
parser.add_argument('--model_path', default='')
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)


parser.add_argument('--conf',type=str,default='')
parser.add_argument('--REC', type=float, default=1)
parser.add_argument('--KL', type=float, default=0.1)
args = parser.parse_args()


if args.conf != "":
    with open(args.conf, 'r') as f:
        conf = json.load(f)
    args.REC = conf["REC"]
    args.KL = conf["KL"]

    #args.model_type = conf["model_type"]

    # print(args)

    

assert args.model_type in ["VAE1", "VAE2", "VAE3", "MD"]
model_dir = args.model_path
latent_dim=8
batch_size = 8
n_frames = 128

is_MD=True if args.model_type == "MD" else False

#initial seed 
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(args.seed)

Enc = model.Encoder(style_dim=len(SPK_LIST), latent_dim=latent_dim, vae_type=args.model_type)
if args.epoch == 0:
    Enc.load_state_dict(torch.load(model_dir+"/final_enc.pt"))
else:
    Enc.load_state_dict(torch.load(model_dir+"/parm/"+str(args.epoch)+"_enc.pt"))
Enc.cuda()
Enc.eval()
if args.model_type == "MD":
    Dec_dict=dict()
    for spk_id in SPK_LIST:
        cur_Dec = model.Decoder(style_dim=len(SPK_LIST), latent_dim=latent_dim, vae_type=args.model_type)
        cur_Dec.cuda()
        cur_Dec.eval()
        if args.epoch == 0:
            cur_Dec.load_state_dict(torch.load(model_dir+"/final_"+spk_id+"_dec.pt"))
        else:
            cur_Dec.load_state_dict(torch.load(model_dir+"/parm/"+str(args.epoch)+"_"+spk_id+"_dec.pt"))
        Dec_dict[spk_id]=cur_Dec
else:
    Dec = model.Decoder(style_dim=len(SPK_LIST), latent_dim=latent_dim, vae_type=args.model_type)
    if args.epoch == 0:
        Dec.load_state_dict(torch.load(model_dir+"/final_dec.pt"))
    else:
        Dec.load_state_dict(torch.load(model_dir+"/parm/"+str(args.epoch)+"_dec.pt"))
    Dec.cuda()
    Dec.eval()



lm = LogManager()
lm.alloc_stat_type_list(["rec_loss", "kl_loss", "total_loss"])

####################################
# do not hard coding 
####################################
coef={"rec": args.REC, "adv": 0.0, "kl": args.KL}
#print("coefficient !!",coef)

# VAE TEST
test_loader = get_loader(SP_DICT_TEST, 1, n_frames=n_frames, shuffle=False, is_MD=is_MD)
for src_idx, A_x, tar_idx in test_loader:
    if is_MD:
        spk_id = SPK_LIST[tar_idx]
        Dec = Dec_group[spk_id]
    
    batch_len = A_x.size()[0]
    A_y = make_spk_vector(tar_idx, TOTAL_SPK_NUM, batch_len, is_MD)
    
    with torch.no_grad():
        one_hot_x = torch.Tensor(make_one_hot_vector(src_idx, TOTAL_SPK_NUM)).cuda()
        one_hot_x = torch.reshape(one_hot_x,(1,TOTAL_SPK_NUM,1))

        one_hot_y = torch.Tensor(make_one_hot_vector(tar_idx, TOTAL_SPK_NUM)).cuda()
        one_hot_y = torch.reshape(one_hot_y,(1,TOTAL_SPK_NUM,1))


        z_mu, z_logvar, A_z = Enc(A_x, A_y, one_hot_x)
        A2A_mu, A2A_logvar, A2A = Dec(A_z, A_y, one_hot_y)

        rec_loss = -calc_gaussprob(A_x, A2A_mu, A2A_logvar)
        kl_loss = calc_kl_vae(z_mu, z_logvar)

        total_loss = coef["rec"] * rec_loss + coef["kl"] * kl_loss
    
    lm.add_torch_stat("rec_loss", rec_loss)
    lm.add_torch_stat("kl_loss", kl_loss)
    lm.add_torch_stat("total_loss", total_loss)

print("TEST EPOCH {}:".format(args.epoch), end=' ')
lm.print_stat()


