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
parser.add_argument('--lr', type=float, default=0)
parser.add_argument('--c_lr', type=float, default=2.5*1e-5)

parser.add_argument('--REC', type=float, default=1)
parser.add_argument('--KL', type=float, default=0.2)
parser.add_argument('--lr_sch',type=str, default='linear15')
parser.add_argument('--epochs',type=int, default=2000)

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

PPG_DICT_TRAIN = {
    spk_id:load_ppg(os.path.join("data_seoulNarrative","train", spk_id)) 
    for spk_id in SPK_LIST
}

PPG_DICT_DEV = {
    spk_id:load_ppg(os.path.join("data_seoulNarrative","dev", spk_id)) 
    for spk_id in SPK_LIST
}

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


coef={ 
    "rec": args.REC, "kl": args.KL , "cyc": args.CC, "si": args.SI, "i": args.I, "li": args.LI, "ac": args.AC, "sc": args.SC
}



print(coef)
print(model_dir)
os.makedirs(model_dir+"/parm", exist_ok=True)

latent_dim=8
## Encoder
Enc = model.Encoder(style_dim=4, latent_dim=latent_dim, vae_type=args.model_type)
# Enc.load_state_dict(torch.load("model/"+args.model_type+"/final_enc.pt"))
Enc.cuda()

lr = 1e-2
c_lr = args.c_lr

lr_sch = args.lr_sch
c_lr_sch = args.lr_sch

if lr_sch[0]=='l':
    lr = 1
    # c_lr = 1
else:
    lr = 1e-2
    # c_lr = lr*0.025

Enc_opt = optim.Adam(Enc.parameters(), lr=lr)
Enc_sch = optim.lr_scheduler.LambdaLR(optimizer=Enc_opt, lr_lambda=lambda epoch: -(1e-2/args.epochs)*epoch+1e-2)

if is_MD:    
    Dec=dict()
    Dec_opt=dict()
    for spk_id in SPK_LIST:
        Dec[spk_id] = model.Decoder(style_dim=4, latent_dim=latent_dim, vae_type=args.model_type)
        # Dec[spk_id].load_state_dict(torch.load("model/"+args.model_type+"/final_"+spk_id+"_dec.pt"))
        Dec[spk_id].cuda()
        Dec_opt[spk_id] = optim.Adam(Dec[spk_id].parameters(), lr=vae_lr)

else:
    Dec = model.Decoder(style_dim=4, latent_dim=latent_dim, vae_type=args.model_type)
    # Dec.load_state_dict(torch.load("model/"+args.model_type+"/final_dec.pt"))
    Dec.cuda()
    Dec_opt = optim.Adam(Dec.parameters(), lr=lr)
    Dec_sch = optim.lr_scheduler.LambdaLR(optimizer=Dec_opt, lr_lambda=lambda epoch: -(1e-2/args.epochs)+1e-2)

torch.save(Enc.state_dict(), os.path.join(model_dir,"final_enc.pt"))
if is_MD:
    for spk_id, cur_dec in Dec.items():
        torch.save(cur_dec.state_dict(), os.path.join(model_dir,"final_"+spk_id+"_dec.pt"))
else:
    torch.save(Dec.state_dict(), os.path.join(model_dir,"final_dec.pt"))



## Classifier
is_conversion = True if (args.AC or args.SC or args.CC or args.GAN) else False
is_classify = True if (args.SI or args.LI) else False
is_adv = True if (args.SI or args.GAN or args.LI) else False
is_revert = True if (args.SC or args.CC) else False
is_pretrain = True if (args.AC) else False


if args.SI:
    spk_C = model.LatentClassifier(latent_dim=latent_dim, label_num=TOTAL_SPK_NUM)
    spk_C.cuda()
    spk_C_opt = optim.Adam(spk_C.parameters(), lr=lr)
    spk_C_sch = optim.lr_scheduler.LambdaLR(optimizer=spk_C_opt, lr_lambda=lambda epoch: c_lr*(-(1e-2/args.epochs)*epoch+1e-2))


if args.LI:
    lang_C = model.LangClassifier(latent_dim=latent_dim, label_num=144)
    lang_C.cuda()
    lang_C_opt = optim.Adam(lang_C.parameters(), lr=c_lr)

if args.AC:
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

if is_pretrain:
    lm = LogManager()
    lm.alloc_stat_type_list(["train_loss", "train_acc", "dev_loss", "dev_acc"])

    if args.AC:
        pretrain_epochs = args.epochs
        batch_size = 8
        print("Train AC")
        for epoch in range(pretrain_epochs):
            lm.init_stat()  
            # Train
            AC.train()
            train_loader = dm.feat_loader_single(SP_DICT_TRAIN, batch_size, shuffle=True)
            for src_idx, x, spk_idxs  in train_loader:
                target_y = dm.make_spk_target(spk_idxs, x.size()[0], is_MD=False)

                pred_y = AC(x)
                spk_loss = nllloss(pred_y, target_y)
                spk_err = calc_err(pred_y, target_y)

                AC_opt.zero_grad()
                spk_loss.backward()
                AC_opt.step()

                lm.add_torch_stat("train_loss", spk_loss)
                lm.add_torch_stat("train_acc", 1.0 - spk_err)

                
            # Dev
            AC.eval()
            dev_loader = dm.feat_loader_single(SP_DICT_DEV, batch_size, shuffle=False)
            for src_idx, x, spk_idxs in dev_loader:
                target_y = dm.make_spk_target(spk_idxs, x.size()[0], is_MD=False)

                pred_y = AC(x)
                spk_loss = nllloss(pred_y, target_y)
                spk_err = calc_err(pred_y, target_y)

                lm.add_torch_stat("dev_loss", spk_loss)
                lm.add_torch_stat("dev_acc", 1.0 - spk_err)
            
            lm.print_stat()
        AC.eval()
    

lm = LogManager()
lm.alloc_stat_type_list(["rec_loss", "kl_loss", "CC_loss", "SI_loss", 
    "I_loss", "LI_loss", "AC_loss", "AC_acc", "SC_loss", "SI_D", "SI_err", "LI_D", "LI_err", "total_loss"])

epochs=args.epochs
for epoch in range(epochs+1):
    # print("EPOCH:   {}  LearningRate:   {}  SpkClassifierLearningRate:   {}".format(epoch, Enc_sch.get_last_lr()[0],spk_C_sch.get_last_lr()[0]))
    print("EPOCH:   {}  LearningRate:   {}".format(epoch, Enc_sch.get_last_lr()[0]))

    
    batch_size = 8
    
    lm.init_stat()  

    start_time = time.time()
    if is_adv:
        Enc.eval()
        set_DEC(Dec, 'eval', is_MD)

        if args.SI:
            spk_C.train()
        if args.LI:
            lang_C.train()
        
        for adv_epoch in range(1):
            adv_loader = dm.feat_loader_single(SP_DICT_TRAIN, batch_size, shuffle=True, ppg_dict=PPG_DICT_TRAIN)
            
            for src_idx, (x, ppg_labs), spk_idxs in adv_loader:
                batch_len = x.size()[0]

                one_hot_x = torch.Tensor(dm.make_one_hot_vector(src_idx, TOTAL_SPK_NUM)).cuda()
                one_hot_x = torch.reshape(one_hot_x,(1,TOTAL_SPK_NUM,1))

                ans_idxs = dm.make_spk_target(src_idx, batch_len, is_MD=False)
                # spk_labs = dm.make_spk_target(spk_idxs, batch_len, is_MD=False)
                y=dm.make_spk_vector(spk_idxs, TOTAL_SPK_NUM, batch_len, is_MD=False)
                one_hot_y = torch.Tensor(dm.make_one_hot_vector(spk_idxs, TOTAL_SPK_NUM)).cuda()
                one_hot_y = torch.reshape(one_hot_y,(1,TOTAL_SPK_NUM,1))

                total_loss_adv = 0.0
                mu, logvar, z = Enc(x, y, one_hot_x)
                cur_opts = []

                batch_len = x.size()[0]
                
                if args.SI:
                    si_z = spk_C(z)
                    # si_loss = nllloss(si_z, spk_labs)
                    # si_err = calc_err(si_z, spk_labs)
                    si_loss = nllloss(si_z, ans_idxs)
                    si_err = calc_err(si_z, ans_idxs)

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
                # print(si_loss)
        if args.SI:
            spk_C.eval()
        if args.LI:
            lang_C.eval()

    # VAE Training
    Enc.train()
    set_DEC(Dec, 'train', is_MD)
    if args.AC:
        AC.train()
    
    train_loader = dm.get_loader(SP_DICT_TRAIN, batch_size, shuffle=True, PPG_DICT=PPG_DICT_TRAIN, is_MD=is_MD)
        
    for src_idx, (A_x, ppg_labs), A_spk_idxs in train_loader:
        if is_MD:
            A_spk_id = SPK_LIST[A_spk_idxs]
        Dec_A = Dec[A_spk_id] if is_MD else Dec
        
        batch_len = A_x.size()[0]
        one_hot_x = torch.Tensor(dm.make_one_hot_vector(src_idx, TOTAL_SPK_NUM)).cuda()
        one_hot_x = torch.reshape(one_hot_x,(1,TOTAL_SPK_NUM,1))

        A_spk_labs = dm.make_spk_target(A_spk_idxs, batch_len, is_MD=is_MD)
        A_y = dm.make_spk_vector(A_spk_idxs, TOTAL_SPK_NUM, batch_len, is_MD=is_MD)
        one_hot_y = torch.Tensor(dm.make_one_hot_vector(A_spk_idxs, TOTAL_SPK_NUM)).cuda()
        one_hot_y = torch.reshape(one_hot_y,(1,TOTAL_SPK_NUM,1))

        total_loss = 0.0
        A_z_mu, A_z_logvar, A_z = Enc(A_x, A_y, one_hot_x)
        
        # Latent Classifier
        if args.SI:
            si_A = spk_C(A_z)
            #  여기
            # si_loss = -1 * nllloss(si_A, A_spk_labs)
            ans_idx = torch.Tensor(src_idx).long().cuda()
            si_loss = nllloss(si_A, ans_idx)
            # si_loss = nllloss(si_A, A_spk_labs)
            total_loss -= coef["si"] * si_loss

        if args.I:
            i_loss = calc_entropy(A_z)
            total_loss += coef["i"] * i_loss
        
        if args.LI:
            li_A = lang_C(A_z)
            li_loss = nllloss(li_A, ppg_labs, is_batch=True)
            total_loss += coef["li"] * li_loss


        # Self-reconstruction
        A2A_mu, A2A_logvar, A2A = Dec_A(A_z, A_y, one_hot_y)

        rec_loss = -calc_gaussprob(A_x, A2A_mu, A2A_logvar)
        kl_loss = calc_kl_vae(A_z_mu, A_z_logvar)
        

        if is_conversion:
            ac_err = 0.0
            ac_loss = 0.0; sc_loss = 0.0; cyc_loss = 0.0

            B_spk_idxs_list = dm.get_all_target_idx(A_spk_idxs, TOTAL_SPK_NUM, is_MD=is_MD)
            for B_spk_idxs in B_spk_idxs_list:
                B_spk_labs = dm.make_spk_target(B_spk_idxs, batch_len, is_MD=is_MD)
                B_y = dm.make_spk_vector(B_spk_idxs, TOTAL_SPK_NUM, batch_len, is_MD=is_MD)

                one_hot_y = torch.Tensor(dm.make_one_hot_vector(B_spk_idxs, TOTAL_SPK_NUM)).cuda()
                one_hot_y = torch.reshape(one_hot_y,(1,TOTAL_SPK_NUM,1))

                if is_MD:
                    B_spk_id = SPK_LIST[B_spk_idxs]
                Dec_B = Dec[B_spk_id] if is_MD else Dec
                A2B_mu, A2B_logvar, A2B = Dec_B(A_z, B_y, one_hot_y)

                # AC
                if args.AC:
                    ac_A2B = AC(A2B)
                    ac_loss += nllloss(ac_A2B, B_spk_labs)
                    ac_err += calc_err(ac_A2B, B_spk_labs)

                    
                 # SC
                if is_revert:
                    A2B_z_mu, A2B_z_logvar, A2B_z = Enc(A2B, B_y)
                    if args.SC:
                        sc_loss += l1loss(A2B_z, A_z)

                    # CYC
                    if args.CC:
                        A2B2A_mu, A2B2A_logvar, _ = Dec_A(A2B_z, A_y)

                        cyc_loss += -calc_gaussprob(A_x, A2B2A_mu, A2B2A_logvar)
                        kl_loss += calc_kl_vae(A2B_z_mu, A2B_z_logvar)
            
            if args.AC:
                ac_err = ac_err/len(B_spk_idxs_list)
            
            total_loss += coef["ac"] * ac_loss + coef["sc"] * sc_loss + coef["cyc"] * cyc_loss
                        
        total_loss += coef["rec"] * rec_loss + coef["kl"] * kl_loss
        
        # Update
        if is_MD:
            opt_list = [dec_opt for dec_opt in Dec_opt.values()]
            opt_list.append(Enc_opt)
        else:
            opt_list = [Enc_opt, Dec_opt]
            if args.AC :
                opt_list.append(AC_opt)
        
        update_parm(opt_list, total_loss)


                        

        # write to log
        lm.add_torch_stat("rec_loss", rec_loss)
        lm.add_torch_stat("kl_loss", kl_loss)
        if args.CC:
            lm.add_torch_stat("CC_loss", cyc_loss)
        if args.SI:
            lm.add_torch_stat("SI_loss", si_loss)
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
    Enc.eval()
    set_DEC(Dec, "eval", is_MD)

    dev_loader = dm.get_loader(SP_DICT_DEV, 1, shuffle=False, PPG_DICT=PPG_DICT_DEV, is_MD=is_MD)

    with torch.no_grad():
        for (A_x, ppg_labs), A_spk_idxs in dev_loader:
            if is_MD:
                A_spk_id = SPK_LIST[A_spk_idxs]
            Dec_A = Dec[A_spk_id] if is_MD else Dec

            batch_len = A_x.size()[0]
            A_spk_labs = dm.make_spk_target(A_spk_idxs, batch_len, is_MD=is_MD)
            A_y = dm.make_spk_vector(A_spk_idxs, TOTAL_SPK_NUM, batch_len, is_MD=is_MD)
            one_hot_y = torch.Tensor(dm.make_one_hot_vector(A_spk_idx, TOTAL_SPK_NUM)).cuda()
            one_hot_y = torch.reshape(one_hot_y,(1,TOTAL_SPK_NUM,1))

            A_z_mu, A_z_logvar, A_z = Enc(A_x, A_y, one_hot_y)
            
            # Latent Classifier
            if args.SI:
                si_A = spk_C(A_z)
                si_loss = -1 * nllloss(si_A, A_spk_labs)

            if args.I:
                i_loss = calc_entropy(A_z)
            
            if args.LI:
                li_A = lang_C(A_z)
                li_loss = nllloss(li_A, ppg_labs, is_batch=True)

            # Self-reconstruction
            A2A_mu, A2A_logvar, A2A = Dec_A(A_z, A_y, one_hot_y)
            
            rec_loss = -calc_gaussprob(A_x, A2A_mu, A2A_logvar)
            kl_loss = calc_kl_vae(A_z_mu, A_z_logvar)

            if is_conversion:
                ac_loss = 0.0; sc_loss = 0.0; cyc_loss = 0.0
                B_spk_idxs_list = dm.get_all_target_idx(A_spk_idxs, TOTAL_SPK_NUM, is_MD=is_MD)
                for B_spk_idxs in B_spk_idxs_list:
                    B_spk_labs = dm.make_spk_target(B_spk_idxs, batch_len, is_MD=is_MD)
                    B_y = dm.make_spk_vector(B_spk_idxs, TOTAL_SPK_NUM, batch_len, is_MD=is_MD)

                    if is_MD:
                        B_spk_id = SPK_LIST[B_spk_idxs]
                    Dec_B = Dec[B_spk_id] if is_MD else Dec
                    A2B_mu, A2B_logvar, A2B = Dec_B(A_z, B_y)
                    # AC
                    if args.AC:
                        ac_A2B = AC(A2B)
                        ac_loss += nllloss(ac_A2B, B_spk_labs)
                        
                    # SC
                    if is_revert:
                        A2B_z_mu, A2B_z_logvar, A2B_z = Enc(A2B, B_y)
                        if args.SC:
                            sc_loss += l1loss(A2B_z, A_z)

                        # CYC
                        if args.CC:
                            A2B2A_mu, A2B2A_logvar, _ = Dec_A(A2B_z, A_y)

                            cyc_loss += -calc_gaussprob(A_x, A2B2A_mu, A2B2A_logvar)
                            kl_loss += calc_kl_vae(A2B_z_mu, A2B_z_logvar)
                                            

            # write to log
            lm.add_torch_stat("rec_loss", rec_loss)
            lm.add_torch_stat("kl_loss", kl_loss)
            if args.CC:
                lm.add_torch_stat("CC_loss", cyc_loss)
            if args.SI:
                lm.add_torch_stat("SI_loss", si_loss)
            if args.I:
                lm.add_torch_stat("I_loss", i_loss)
            if args.LI:
                lm.add_torch_stat("LI_loss", li_loss)
            if args.AC:
                lm.add_torch_stat("AC_loss", ac_loss)
            if args.SC:
                lm.add_torch_stat("SC_loss", sc_loss)
            lm.add_torch_stat("total_loss", total_loss)

    print("DEV:", end=' ')
    lm.print_stat()
    end_time = time.time()

    total_time += (end_time - start_time)

    print(".....................")

    Enc_sch.step()
    Dec_sch.step()
    
    if args.SI:
        spk_C_sch.step()
    
    # if epoch % 10 == 0:
        ### check min loss
    cur_loss = lm.get_stat("total_loss")
    
    if np.isnan(cur_loss):
        print("Nan at",epoch)
        break


    if min_dev_loss > cur_loss:
        min_dev_loss = cur_loss
        min_epoch = epoch

    ### Parmaeter save
    torch.save(Enc.state_dict(), os.path.join(model_dir,"parm",str(epoch)+"_enc.pt"))
    if is_MD:
        for spk_id, cur_dec in Dec.items():
            torch.save(cur_dec.state_dict(), os.path.join(model_dir,"parm",str(epoch)+"_"+spk_id+"_dec.pt"))
    else:
        torch.save(Dec.state_dict(), os.path.join(model_dir,"parm",str(epoch)+"_dec.pt"))
    
print("***********************************")
print("Model name:",model_dir.split("/")[-1])
print("TIME PER EPOCH:",total_time/epochs)
print("Final Epoch:",min_epoch, min_dev_loss)
print("***********************************")

# min_epoch = epochs

os.system("cp "+os.path.join(model_dir,"parm",str(min_epoch)+"_enc.pt")+" "+os.path.join(model_dir,"final_enc.pt"))
if is_MD:
    for spk_id, cur_dec in Dec.items():
        os.system("cp "+os.path.join(model_dir,"parm",str(min_epoch)+"_"+spk_id+"_dec.pt")+" "+os.path.join(model_dir,"final_"+spk_id+"_dec.pt"))
else:
    os.system("cp "+os.path.join(model_dir,"parm",str(min_epoch)+"_dec.pt")+" "+os.path.join(model_dir,"final_dec.pt"))
