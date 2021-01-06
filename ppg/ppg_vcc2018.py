import os, sys
import net
import torch
import pickle as pk
import numpy as np
from tools import kaldi_manager as km
from tools import data_manager as dm

model_path="model/timit_sp_ppg_mono"
model = net.BiGRU_HMM(feat_dim=36, hidden_dim=256, num_layers=5, dropout=0.2, output_dim=144)
model.load_state_dict(torch.load(model_path+"/final.pt"))
model.cuda()
model.eval()

with open(model_path+"/prior.pk", 'rb') as f:
    prior = pk.load(f)

feat_root = "../data"

# Train
train_path = os.path.join(feat_root, "train")
train_dict = dict()
for root, dirs, files in os.walk(train_path):
    for spk_id in dirs:
        sp_path = os.path.join(train_path,spk_id,"feats.p")
        with open(sp_path, 'rb') as f:
            sp_list, sp_m, sp_s, logf0_m, logf0_s = pk.load(f)
        ppg_list = []
        for feat_mat in sp_list:
            x = torch.Tensor(feat_mat).float().cuda().permute(1,0)
            with torch.no_grad():
                result = model([x])
            result = result.detach().cpu().numpy()
            ppg = result - prior
            ppg_list.append(ppg.T)
        ppg_path = os.path.join(train_path,spk_id,"ppg36.p")
        with open(ppg_path, 'wb') as f:
            pk.dump(ppg_list, f)

# Dev
dev_path = os.path.join(feat_root, "dev")
dev_dict = dict()
for root, dirs, file_list in os.walk(dev_path):
    for spk_id in dirs:
        ppg_list = [] 
        for _, _, dev_list in os.walk(dev_path+"/"+spk_id):
            for file_id in dev_list:
                dev_id = file_id.split(".")[0]
                if dev_id == "ppg36":
                    continue
                sp_path = os.path.join(dev_path,spk_id,dev_id+".p")
                with open(sp_path, 'rb') as f:
                    feat_mat, f0, ap = pk.load(f)

                x = torch.Tensor(feat_mat).float().cuda().permute(1,0)
                with torch.no_grad():
                    result = model([x])
                result = result.detach().cpu().numpy()
                ppg = result - prior
                ppg_list.append(ppg.T)

            ppg_path = os.path.join(dev_path,spk_id,"ppg36.p")
            with open(ppg_path, 'wb') as f:
                pk.dump(ppg_list, f)