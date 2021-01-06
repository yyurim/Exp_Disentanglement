import json
import os, sys
import argparse
import numpy as np
import time
import torch
import torch.optim as optim
from loss import LogManager, calc_gaussprob, calc_kl_vae
import pickle
import model
from data_manager import get_loader, make_spk_vector, make_one_hot_vector
from itertools import combinations

# get data
#   train, dev : coded sp, f0, ap
#   test : sp, coded sp, f0, ap
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

# backpropagation
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
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--model_dir', default='')
parser.add_argument('--lr', type=float, default=0)

parser.add_argument('--lr_sch',type=str, default='linear15')
parser.add_argument('--epochs',type=int, default=2000)

args = parser.parse_args()
assert args.model_type in ["VAE1", "VAE2", "VAE3", "MD"]
assert args.lr_sch in ['expo1','expo2','expo3','linear11','linear12','linear13',\
'linear21','linear22','linear23',\
'linear14','linear15','linear16',\
'expo4','expo5','expo6']


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
#   Hyperparameters
#       do not hard coding
#       set with conf file
######################################
coef={"rec": args.REC, "adv": 0.0, "kl": args.KL}
print("coefficient !!",coef)
print(model_dir)
os.makedirs(model_dir+"/parm", exist_ok=True)latent_dim=8

is_MD=True if args.model_type == "MD" else False
lr_sch = args.lr_sch
lr = 1

epochs = args.epochs

total_time = 0
min_dev_loss = 9999999999999999
min_epoch = 0
d_epoch = 1

batch_size = 8

n_frames = 128
latent_dim=8


######################################
#   Model initialize
######################################

# Encoder
Enc = model.Encoder(style_dim=TOTAL_SPK_NUM, latent_dim=latent_dim, vae_type=args.model_type)
Enc.cuda()
Enc_opt = optim.Adam(Enc.parameters(), lr=lr)
# linear [1e-2 , 0] epoch 2000
Enc_sch = optim.lr_scheduler.LambdaLR(optimizer=Enc_opt, lr_lambda=lambda epoch: -(1e-2/args.epochs)*epoch+1e-2)
print(calc_parm_num(Enc))


# Decoder
if is_MD:    
    Dec_group=dict()
    Dec_opt_group=dict()
    Dec_sch_group=dict()
    for spk_id in SPK_LIST:
        Dec_group[spk_id] = model.Decoder(style_dim=TOTAL_SPK_NUM, latent_dim=latent_dim, vae_type=args.model_type)
        Dec_group[spk_id].cuda()
        Dec_opt_group[spk_id] = optim.Adam(Dec_group[spk_id].parameters(), lr=lr)
        Dec_sch_group[spk_id] = optim.lr_scheduler.ExponentialLR(Dec_opt_group[spk_id], 0.9)
        
else:
    Dec = model.Decoder(style_dim=TOTAL_SPK_NUM, latent_dim=latent_dim, vae_type=args.model_type)
    Dec.cuda()
    Dec_opt = optim.Adam(Dec.parameters(), lr=lr)
    Dec_sch = optim.lr_scheduler.LambdaLR(optimizer=Dec_opt, lr_lambda=lambda epoch: -(1e-2/args.epochs)*epoch+1e-2)


print(Enc)
print(Dec)

# 8 16
# (0-499) (500-999)
print("Training Settings")
print("LR",lr)
print("Number of epochs",epochs)
print(".....................")
lm = LogManager()
lm.alloc_stat_type_list(["rec_loss", "kl_loss", "total_loss"])


for epoch in range(epochs+1):
    print("EPOCH:   {}  LearningRate:   {}".format(epoch, Enc_sch.get_last_lr()[0]))
    lm.init_stat()  

    start_time = time.time()

    # VAE Training
    Enc.train()
    if is_MD:
        for dec in Dec_group.values():
            dec.train()
    else:
        Dec.train()
    
    train_loader = get_loader(SP_DICT_TRAIN, batch_size, n_frames=n_frames, shuffle=True, is_MD=is_MD)

    for src_idx, A_x, spk_idx in train_loader:
        #   src_idx : source idx
        #   A_x : coded_sp
        #   spk_idx : target idx

        if is_MD:
            spk_id = SPK_LIST[spk_idx]
            Dec = Dec_group[spk_id]
            Dec_opt = Dec_opt_group[spk_id]
            Dec_sch = Dec_sch_group[spk_id]
        
        batch_len = A_x.size()[0]

        #   one hot as condition
        one_hot_x = torch.Tensor(make_one_hot_vector(src_idx, TOTAL_SPK_NUM)).cuda()
        one_hot_x = torch.reshape(one_hot_x,(1,TOTAL_SPK_NUM,1))

        #   target vector for latent
        A_y = make_spk_vector(spk_idx, TOTAL_SPK_NUM, batch_len, is_MD)

        #   one hot as condition
        one_hot_y = torch.Tensor(make_one_hot_vector(spk_idx, TOTAL_SPK_NUM)).cuda()
        one_hot_y = torch.reshape(one_hot_y,(1,TOTAL_SPK_NUM,1))
        
        # VAE
        z_mu, z_logvar, A_z = Enc(A_x, A_y, one_hot_x)
        A2A_mu, A2A_logvar, A2A = Dec(A_z, A_y, one_hot_y)

        # calculate loss
        #       ratio >> reconstruction : kl = coef["rec"] : coef["kl"]
        rec_loss = -calc_gaussprob(A_x, A2A_mu, A2A_logvar)
        kl_loss = calc_kl_vae(z_mu, z_logvar)

        total_loss = coef["rec"] * rec_loss + coef["kl"] * kl_loss
       
        #    backpropagation
        update_parm([Enc_opt, Dec_opt], total_loss, epoch)

        # write to log
        lm.add_torch_stat("rec_loss", rec_loss)
        lm.add_torch_stat("kl_loss", kl_loss)
        lm.add_torch_stat("total_loss", total_loss)

    print("Train:", end=' ')
    
    lm.print_stat()

    # VAE Evaluation
    lm.init_stat()
    Enc.eval()
    if is_MD:
        for dec in Dec_group.values():
            dec.eval()
    else:
        Dec.eval()
    
    dev_loader = get_loader(SP_DICT_DEV, 1, n_frames=n_frames, shuffle=False, is_MD=is_MD)

    for src_idx, A_x, spk_idx in dev_loader:
        # print("target   !!!",spk_idx)
        if is_MD:
            spk_id = SPK_LIST[spk_idx]
            Dec = Dec_group[spk_id]
        
        batch_len = A_x.size()[0]
        one_hot_x = torch.Tensor(make_one_hot_vector(src_idx, TOTAL_SPK_NUM)).cuda()
        one_hot_x = torch.reshape(one_hot_x,(1,TOTAL_SPK_NUM,1))

        A_y = make_spk_vector(spk_idx, TOTAL_SPK_NUM, batch_len, is_MD)
        one_hot_y = torch.Tensor(make_one_hot_vector(spk_idx, TOTAL_SPK_NUM)).cuda()
        one_hot_y = torch.reshape(one_hot_y,(1,4,1))
        
        with torch.no_grad():
            z_mu, z_logvar, A_z = Enc(A_x, A_y,one_hot_x)
            A2A_mu, A2A_logvar, A2A = Dec(A_z, A_y,one_hot_y)

            rec_loss = -calc_gaussprob(A_x, A2A_mu, A2A_logvar)
            kl_loss = calc_kl_vae(z_mu, z_logvar)

            total_loss = coef["rec"] * rec_loss + coef["kl"] * kl_loss
        
        lm.add_torch_stat("rec_loss", rec_loss)
        lm.add_torch_stat("kl_loss", kl_loss)
        lm.add_torch_stat("total_loss", total_loss)
    
    print("DEV:", end=' ')
    lm.print_stat()
    end_time = time.time()

    total_time += (end_time - start_time)

    print(".....................")

    # lr scheduling
    Enc_sch.step()
    Dec_sch.step()
    
    cur_loss = lm.get_stat("total_loss")
    if np.isnan(cur_loss):
        print("Nan at",epoch)
        break

    # get epoch with minimum dev total loss 
    if min_dev_loss > cur_loss:
        min_dev_loss = cur_loss
        min_epoch = epoch


    # save model
    torch.save(Enc.state_dict(), os.path.join(model_dir,"parm",str(epoch)+"_enc.pt"))

    if args.model_type == "MD":
        for spk_id, Dec in Dec_group.items():
            torch.save(Dec.state_dict(), os.path.join(model_dir,"parm",str(epoch)+"_"+spk_id+"_dec.pt"))
    else:
        torch.save(Dec.state_dict(), os.path.join(model_dir,"parm",str(epoch)+"_dec.pt"))


print("***********************************")
print("Model name:",model_dir.split("/")[-1])
print("TIME PER EPOCH:",total_time/(epochs+1))
print("Final Epoch:",min_epoch, min_dev_loss)
print("***********************************")

# save minimum dev total loss model
os.system("cp "+os.path.join(model_dir,"parm",str(min_epoch)+"_enc.pt")+" "+os.path.join(model_dir,"final_enc.pt"))
if args.model_type == "MD":
    for spk_id, Dec in Dec_group.items():
        os.system("cp "+os.path.join(model_dir,"parm",str(min_epoch)+"_"+spk_id+"_dec.pt")+" "+os.path.join(model_dir,"final_"+spk_id+"_dec.pt"))
else:
    os.system("cp "+os.path.join(model_dir,"parm",str(min_epoch)+"_dec.pt")+" "+os.path.join(model_dir,"final_dec.pt"))
print("\nsaved optimal model? >> ", os.path.isfile(os.path.join(model_dir,"final_enc.pt")))

