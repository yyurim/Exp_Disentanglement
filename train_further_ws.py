import os, sys
import argparse
import numpy as np
import time
import torch
import torch.optim as optim
from loss import LogManager, calc_gaussprob, calc_kl_vae, nllloss, calc_entropy, calc_err, l1loss, calc_entropy_log
import pickle
import model_exp as model
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

def update_parm(opt_list, loss):
    for opt in opt_list:
        opt.zero_grad()
    loss.backward()
    for opt in opt_list:
        opt.step()


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
parser.add_argument('--model_type', type=str) # VAE3 MD
parser.add_argument('--SI', type=int, default=0)
parser.add_argument('--I', type=int, default=0)
parser.add_argument('--LI', type=int, default=0)
parser.add_argument('--AC', type=int, default=0)
parser.add_argument('--SC', type=int, default=0)
parser.add_argument('--CC', type=int, default=0)
parser.add_argument('--GAN', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--conf', type=str, default="")
parser.add_argument('--model_dir', default='')
parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--c_lr', type=float, default=2.5*1e-5)

parser.add_argument('--REC', type=float, default=1)
parser.add_argument('--KL', type=float, default=0.2)
parser.add_argument('--lr_sch',type=str, default='linear15')
parser.add_argument('--epochs',type=int, default=2000)

parser.add_argument('--baseline',type=str, default='')
parser.add_argument('--disentanglement', type=str, default='')

args = parser.parse_args()
assert args.model_type in ["VAE1", "VAE2", "VAE3", "MD"]

is_MD=True if args.model_type=="MD" else False

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(args.seed)

if args.conf != "":
    with open(args.conf, 'r') as f:
        conf = json.load(f)
    args.REC = conf["REC"]
    args.KL = conf["KL"]
    args.SI = conf["SI"]
    args.I = conf["I"]
    args.LI = conf["LI"]
    args.AC = conf["AC"]
    args.CC = conf["CC"]
    args.SC = conf["SC"]
    args.model_type = conf["model_type"]

    print(args)


# Data load
SPK_LIST = ['F1','M1','F2','M2']
TOTAL_SPK_NUM = len(SPK_LIST)

# PPG_DICT_TRAIN = {
#     spk_id:load_ppg(os.path.join("data_seoulNarrative","train", spk_id)) 
#     for spk_id in SPK_LIST
# }
PPG_DICT_TRAIN = None
# PPG_DICT_DEV = {
#     spk_id:load_ppg(os.path.join("data_seoulNarrative","dev", spk_id)) 
#     for spk_id in SPK_LIST
# }
PPG_DICT_DEV = None
SP_DICT_TRAIN = {
    spk_id:load_sp(os.path.join("data_seoulNarrative","train", spk_id)) 
    for spk_id in SPK_LIST
}

# print(SP_DICT_TRAIN)
# exit()

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


coef={ 
    "rec": args.REC, "kl": args.KL , "cyc": args.CC, "si": args.SI, "i": args.I, "li": args.LI, "ac": args.AC, "sc": args.SC
}


print(coef)
print(model_dir)
os.makedirs(model_dir+"/parm", exist_ok=True)

latent_dim=8

lr = 1
c_lr = args.c_lr


batch_size = 8



## Classifier
is_conversion = True if (args.AC or args.SC or args.CC or args.GAN) else False
is_classify = True if (args.SI or args.LI) else False
is_adv = True if (args.SI or args.GAN or args.LI) else False
is_revert = True if (args.SC or args.CC) else False
is_pretrain = True if (args.AC) else False


vae = model.VAE(style_dim=TOTAL_SPK_NUM, latent_dim=latent_dim, vae_type=args.model_type)
vae.cuda()
vae_opt = optim.Adam(vae.parameters(), lr=lr)
vae_sch = optim.lr_scheduler.LambdaLR(optimizer=vae_opt, lr_lambda=lambda epoch: -(1e-2/args.epochs)*epoch+1e-2)
print(calc_parm_num(vae))
print(vae)

pretrained_si = False
if os.path.isfile("pretrain_SI/500_si.pt"):
    pretrained_si = True

if args.SI:
    spk_C = model.LatentClassifier(latent_dim=latent_dim, label_num=TOTAL_SPK_NUM)
    if pretrained_si:
        spk_C.load_state_dict(torch.load("pretrain_SI/500_si.pt"))
    spk_C.cuda()
    spk_C_opt = optim.Adam(spk_C.parameters(), lr=c_lr)
    spk_C_sch = optim.lr_scheduler.LambdaLR(optimizer=spk_C_opt, lr_lambda=lambda epoch: c_lr*(-(1e-2/(args.epochs+500+1))*epoch+1e-2))

    print(calc_parm_num(spk_C))
    print(spk_C)

if args.LI:
    lang_C = model.LangClassifier(latent_dim=latent_dim, label_num=144)
    lang_C.cuda()
    lang_C_opt = optim.Adam(lang_C.parameters(), lr=c_lr)
    

if args.AC:
    ac = model.DataClassifier(latent_dim=latent_dim, label_num=TOTAL_SPK_NUM)
    ac.load_state_dict(torch.load('pretrain_AC/parm/499_ac.pt'))
    ac.cuda()
    ac.eval()

    print(calc_parm_num(ac))
    print(ac)


torch.save(vae.state_dict(), os.path.join(model_dir,"final_{}.pt".format(args.disentanglement)))

# 8 16
# (0-499) (500-999)
total_time = 0

min_dev_loss = 9999999999999999
min_epoch = 0
d_epoch = 1

if pretrained_si is False:
    pre_vae = model.VAE(style_dim=TOTAL_SPK_NUM, latent_dim=latent_dim, vae_type=args.model_type)
    pre_vae.load_state_dict(torch.load(args.baseline))
    pre_vae.cuda()
    pre_vae.eval()

    lm = LogManager()
    lm.alloc_stat_type_list(["train_loss", "train_acc", "dev_loss", "dev_acc"])

    for epoch in range(500+1):
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

        spk_C.eval()
        dev_loader = dm.feat_loader_single(SP_DICT_TRAIN, batch_size, shuffle=True)      
        for self_idx, coded_mcep in dev_loader:
            
            one_hot_self = dm.make_spk_vector(self_idx, TOTAL_SPK_NUM, batch_size, is_MD)
            # one_hot_y = dm.make_spk_vector(tar_idx, TOTAL_SPK_NUM, batch_size, is_MD)
            
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

            lm.add_torch_stat("dev_loss", si_loss)
            lm.add_torch_stat("dev_acc", 1-si_err)
        
        print("SI Epoch: ",epoch,end=' / ')
        lm.print_stat()
        spk_C_sch.step()

    os.makedirs("pretrain_SI",exist_ok=True)
    torch.save(spk_C.state_dict(), "pretrain_SI/500_si.pt")



lm = LogManager()
lm.alloc_stat_type_list(["rec_loss", "kl_loss", "CC_loss", "SI_loss", 
    "I_loss", "LI_loss", "AC_loss", "AC_acc","SC_loss", "SI_D", "SI_err", "LI_D", "LI_err", "total_loss"])

epochs=args.epochs

for epoch in range(epochs+1):
    print("EPOCH:   {}  LearningRate:   {}".format(epoch, vae_sch.get_last_lr()[0]),end='   ')
    if args.SI:
        print("SILearningRate:   {}".format(spk_C_sch.get_last_lr()[0]),end='   ')
    print()
    
    batch_size = 8
    
    lm.init_stat()  

    start_time = time.time()

    if is_adv:
        vae.eval()

        if args.SI:
            spk_C.train()
        if args.LI:
            lang_C.train()
        
        adv_loader = dm.feat_loader_single(SP_DICT_TRAIN, batch_size, shuffle=True, ppg_dict=PPG_DICT_TRAIN)

        # for self_idx, (coded_mcep, ppg_labs) in adv_loader:
        for self_idx, coded_mcep in adv_loader:
            one_hot_self = dm.make_spk_vector(self_idx, TOTAL_SPK_NUM, batch_size, is_MD)
    
            z_mu, z_logvar, z, _,_,_ = vae(x=coded_mcep,one_hot_src=one_hot_self, one_hot_tar=one_hot_self)

            total_loss_adv = 0.0

            cur_opts = []

            if args.SI:
                self_vec = dm.make_spk_target(self_idx, batch_size, is_MD=is_MD)
                predicted_self = spk_C(z)
                si_loss = nllloss(predicted_self, self_vec)
                si_err = calc_err(predicted_self, self_vec)

                total_loss_adv += si_loss
                cur_opts.append(spk_C_opt)

            if args.LI:
                li_z = lang_C(z)
                li_loss = nllloss(li_z, ppg_labs, is_batch=True)
                li_err = calc_err(li_z, ppg_labs, is_batch=True)

                total_loss_adv += li_loss
                cur_opts.append(lang_C_opt)


            update_parm(cur_opts, total_loss_adv)


            if args.SI:
                lm.add_torch_stat("SI_D", si_loss)
                lm.add_torch_stat("SI_err", si_err)

            if args.LI:
                lm.add_torch_stat("LI_D", li_loss)
                lm.add_torch_stat("LI_err", li_err)

        if args.SI:
            spk_C.eval()
        if args.LI:
            lang_C.eval()

    # VAE Training
    vae.train()
    
    train_loader = dm.feat_loader_pair(SP_DICT_TRAIN, batch_size, shuffle=True, is_MD=is_MD, ppg_dict=PPG_DICT_TRAIN)

    # for self_idx, (self_coded_mcep, self_ppg_labs), tar_idxs, tar_coded_mceps in train_loader:
    for self_idx, self_coded_mcep, tar_idxs, tar_coded_mceps in train_loader:
        total_loss = 0.0         

        # VAE
        one_hot_self = dm.make_spk_vector(self_idx, TOTAL_SPK_NUM, batch_size, is_MD)

        self_z_mu, self_z_logvar, self_z, x_prime_mu, x_prime_logvar, x_prime = vae(x=self_coded_mcep,one_hot_src=one_hot_self, one_hot_tar=one_hot_self)

        rec_loss = -calc_gaussprob(self_coded_mcep, x_prime_mu, x_prime_logvar)
        kl_loss = calc_kl_vae(self_z_mu, self_z_logvar)      

        total_loss += coef["rec"] * rec_loss + coef["kl"] * kl_loss

        # Latent Classifier
        if args.SI:
            self_vec = dm.make_spk_target(self_idx, batch_size, is_MD=is_MD)
            predicted_self = spk_C(self_z)
            si_loss = -1 * nllloss(predicted_self, self_vec)
            # si_err = calc_err(predicted_x, x)

            total_loss += coef["si"] * si_loss
        
        # ACVAE
        if args.AC:
            ac_loss = 0.0
            ac_err = 0.0

            for i in range(TOTAL_SPK_NUM - 1):
                tar_idx = tar_idxs[i]
                tar_coded_mcep = tar_coded_mceps[i]

                one_hot_tar = dm.make_spk_vector(tar_idx, TOTAL_SPK_NUM, batch_size, is_MD)

                y_z_mu, y_z_logvar, y_z, y_prime_mu, y_prime_logvar, y_prime = vae(x=self_coded_mcep ,one_hot_src=one_hot_self, one_hot_tar=one_hot_tar)

                y = dm.make_spk_target(tar_idx, batch_size, is_MD=is_MD)
            
                predict_y_e2e = ac(y_prime)
                ac_loss += nllloss(predict_y_e2e, y)
                ac_err += calc_err(predict_y_e2e, y)

            ac_loss /= (TOTAL_SPK_NUM - 1)
            ac_err /= (TOTAL_SPK_NUM - 1)
            
            total_loss += coef["ac"] * ac_loss

        # semantic consistency
        if args.SC:
            sc_loss = 0.0

            for i in range(TOTAL_SPK_NUM - 1):
                tar_idx = tar_idxs[i]

                one_hot_tar = dm.make_spk_vector(tar_idx, TOTAL_SPK_NUM, batch_size, is_MD)

                x_z, y_z = vae(x=self_coded_mcep ,one_hot_src=one_hot_self, one_hot_tar=one_hot_tar, is_SC=True)

                sc_loss += l1loss(x_z, y_z)
            
            sc_loss /= (TOTAL_SPK_NUM - 1)

            total_loss += coef["sc"] * sc_loss

        # I
        if args.I:
            i_loss = calc_entropy(self_z)
            total_loss += coef["i"] * i_loss

        # language info in latent space
        if args.LI:
            lang_info = lang_C(self_z)
            li_loss = nllloss(lang_info, ppg_labs,is_batch=True)
            total_loss += coef["li"] * li_loss



        # Update
        if is_MD:
            opt_list.append(vae_opt)
        else:
            opt_list = [vae_opt]
            if args.SI :
                opt_list.append(spk_C_opt)
        
        update_parm(opt_list, total_loss)

        # write to log
        lm.add_torch_stat("rec_loss", rec_loss)
        lm.add_torch_stat("kl_loss", kl_loss)
        if args.CC:
            lm.add_torch_stat("CC_loss", cyc_loss)
        if args.SI:
            lm.add_torch_stat("SI_loss", -si_loss)
        if args.I:
            lm.add_torch_stat("I_loss", i_loss)
        if args.LI:
            lm.add_torch_stat("LI_loss", li_loss)
        if args.AC:
            lm.add_torch_stat("AC_loss", ac_loss)
            lm.add_torch_stat("AC_acc", 1-ac_err)
        if args.SC:
            lm.add_torch_stat("SC_loss", sc_loss)
        lm.add_torch_stat("total_loss", total_loss)


    print("Train:", end=' ')
    lm.print_stat()

    # VAE Evaluation
    lm.init_stat()
    vae.eval()
    
    dev_loader = dm.feat_loader_pair(SP_DICT_TRAIN, batch_size, shuffle=True, is_MD=is_MD, ppg_dict=PPG_DICT_DEV)

    with torch.no_grad():
        # for self_idx, (self_coded_mcep, self_ppg_labs), tar_idxs, tar_coded_mceps  in dev_loader: 
        for self_idx, self_coded_mcep, tar_idxs, tar_coded_mceps  in dev_loader: 
            total_loss = 0.0         

            # VAE
            one_hot_self = dm.make_spk_vector(self_idx, TOTAL_SPK_NUM, batch_size, is_MD)

            self_z_mu, self_z_logvar, self_z, x_prime_mu, x_prime_logvar, x_prime = vae(x=self_coded_mcep,one_hot_src=one_hot_self, one_hot_tar=one_hot_self)

            rec_loss = -calc_gaussprob(self_coded_mcep, x_prime_mu, x_prime_logvar)
            kl_loss = calc_kl_vae(self_z_mu, self_z_logvar)      

            total_loss += coef["rec"] * rec_loss + coef["kl"] * kl_loss

            # Latent Classifier
            if args.SI:
                self_vec = dm.make_spk_target(self_idx, batch_size, is_MD=is_MD)
                predicted_self = spk_C(self_z)
                si_loss = -1 * nllloss(predicted_self, self_vec)

                total_loss += coef["si"] * si_loss

            if args.AC:

                # ACVAE
                ac_loss = 0.0
                ac_err = 0.0

                for i in range(TOTAL_SPK_NUM - 1):
                    tar_idx = tar_idxs[i]
                    tar_coded_mcep = tar_coded_mceps[i]

                    one_hot_tar = dm.make_spk_vector(tar_idx, TOTAL_SPK_NUM, batch_size, is_MD)

                    y_z_mu, y_z_logvar, y_z, y_prime_mu, y_prime_logvar, y_prime = vae(x=self_coded_mcep ,one_hot_src=one_hot_self, one_hot_tar=one_hot_tar)

                    y = dm.make_spk_target(tar_idx, batch_size, is_MD=is_MD)
                
                    predict_y_e2e = ac(y_prime)
                    ac_loss += nllloss(predict_y_e2e, y)
                    ac_err += calc_err(predict_y_e2e, y)

                ac_loss /= (TOTAL_SPK_NUM - 1)
                ac_err /= (TOTAL_SPK_NUM - 1)
                
                total_loss += coef["ac"] * ac_loss

            if args.SC:
                sc_loss = 0.0

                for i in range(TOTAL_SPK_NUM - 1):
                    tar_idx = tar_idxs[i]

                    one_hot_tar = dm.make_spk_vector(tar_idx, TOTAL_SPK_NUM, batch_size, is_MD)

                    x_z, y_z = vae(x=self_coded_mcep ,one_hot_src=one_hot_self, one_hot_tar=one_hot_tar, is_SC=True)

                    sc_loss += l1loss(x_z, y_z)
                
                sc_loss /= (TOTAL_SPK_NUM - 1)

                total_loss += coef["sc"] * sc_loss

            # I
            if args.I:
                i_loss = calc_entropy(self_z)
                total_loss += coef["i"]*i_loss

            # language info in latent space
            if args.LI:
                lang_info = lang_C(self_z)
                li_loss = nllloss(lang_info, ppg_labs, is_batch=True)
                total_loss += coef["li"] * li_loss

                    
            # write to log
            lm.add_torch_stat("rec_loss", rec_loss)
            lm.add_torch_stat("kl_loss", kl_loss)
            if args.CC:
                lm.add_torch_stat("CC_loss", cyc_loss)
            if args.SI:
                lm.add_torch_stat("SI_loss", -si_loss)
            if args.I:
                lm.add_torch_stat("I_loss", i_loss)
            if args.LI:
                lm.add_torch_stat("LI_loss", li_loss)
            if args.AC:
                lm.add_torch_stat("AC_loss", ac_loss)
                lm.add_torch_stat("AC_acc", 1-ac_err)
            if args.SC:
                lm.add_torch_stat("SC_loss", sc_loss)
            lm.add_torch_stat("total_loss", total_loss)

        

    print("DEV:", end=' ')
    lm.print_stat()
    end_time = time.time()

    total_time += (end_time - start_time)

    print(".....................")

    vae_sch.step()

    if args.SI:
        spk_C_sch.step()
    
    
    cur_loss = lm.get_stat("total_loss")
    
    if np.isnan(cur_loss):
        print("Nan at",epoch)
        break


    if min_dev_loss > cur_loss:
        min_dev_loss = cur_loss
        min_epoch = epoch

    ### Parmaeter save
    torch.save(vae.state_dict(), os.path.join(model_dir,"parm",str(epoch)+"_{}.pt".format(args.disentanglement)))
    
print("***********************************")
print("Model name:",model_dir.split("/")[-1])
print("TIME PER EPOCH:",total_time/epochs)
print("Final Epoch:",min_epoch, min_dev_loss)
print("***********************************")

# min_epoch = epochs

os.system("cp "+os.path.join(model_dir,"parm",str(min_epoch)+"_{}.pt".format(args.disentanglement))+" "+os.path.join(model_dir,"final_{}.pt".format(args.disentanglement)))

