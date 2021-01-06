import os, sys
import model
import argparse
import soundfile
import torch
import numpy as np
import pickle

from speech_tools import world_decode_mc, world_speech_synthesis

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def make_one_hot_vector(spk_idx, spk_num):
    vec = np.zeros(spk_num)
    vec[spk_idx] = 1.0
    return vec

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add(mu)

#spk_list = ["VCC2SF1", "VCC2SF2", "VCC2SM1", "VCC2SM2"]
spk_list = ['F1','M1','F2','M2']
#spk_list = ['F1', 'M1', 'F2', 'M2', 'M3', 'F3', 'M4', 'F4', 'F5', 'M5', 'F6', 'M6', 'M7', 'F7', 'F8', 'M8', 'M9', 'F9', 'M10', 'F10', 'M11', 'F11', 'M12', 'F12', 'F13', 'M13', 'F14', 'M14', 'M15', 'F15', 'F16', 'M16', 'M17', 'F17', 'F18', 'M18', 'F19', 'M19', 'M20', 'F20', 'F21', 'M21', 'F22', 'M22', 'M23', 'F23', 'F24', 'M24', 'M25', 'F25', 'F26', 'M26', 'M27', 'F27', 'F28', 'M28', 'F29', 'M29', 'F30', 'M30', 'F31', 'M31', 'F32', 'M32', 'M33', 'F33', 'M34', 'F34', 'M35', 'F35', 'F36', 'M36', 'M37', 'F37', 'M38', 'F38', 'M39', 'F39', 'F40', 'M40', 'F41', 'M41', 'M42', 'F42', 'F43', 'M43', 'F44', 'M44', 'F45', 'M45', 'F46', 'M46', 'F47', 'M47', 'M48', 'F48', 'F49', 'M49', 'F50', 'M50', 'M51', 'F51', 'F52', 'M52', 'F53', 'M53', 'F54', 'M54', 'M55', 'F55', 'M56', 'F56']
TOTAL_SPK_NUM = len(spk_list)

SPK_DICT = {
    spk_idx:spk_id 
    for spk_idx, spk_id in enumerate(spk_list)
}
VEC_DICT = {
    spk_id:[make_one_hot_vector(spk_idx, len(spk_list))]
    for spk_idx, spk_id in SPK_DICT.items()
}

parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='test')
parser.add_argument('--model_type', default='')
parser.add_argument('--model_path', default='')
parser.add_argument('--convert_path', default='')
parser.add_argument('--epoch', type=int, default=0)
args = parser.parse_args()

mode = args.mode
assert mode in ['train4eval','test']

model_dir = args.model_path
if args.convert_path=='':
    convert_path = "result/"+args.model_path.split("/")[-1]
else:
    convert_path = args.convert_path
    
latent_dim=8

Enc = model.Encoder(style_dim=len(spk_list), latent_dim=latent_dim, vae_type=args.model_type)
if args.epoch == 0:
    Enc.load_state_dict(torch.load(model_dir+"/final_enc.pt"))
else:
    Enc.load_state_dict(torch.load(model_dir+"/parm/"+str(args.epoch)+"_enc.pt"))
Enc.cuda()
Enc.eval()
if args.model_type == "MD":
    Dec_dict=dict()
    for spk_id in spk_list:
        cur_Dec = model.Decoder(style_dim=len(spk_list), latent_dim=latent_dim, vae_type=args.model_type)
        cur_Dec.cuda()
        cur_Dec.eval()
        if args.epoch == 0:
            cur_Dec.load_state_dict(torch.load(model_dir+"/final_"+spk_id+"_dec.pt"))
        else:
            cur_Dec.load_state_dict(torch.load(model_dir+"/parm/"+str(args.epoch)+"_"+spk_id+"_dec.pt"))
        Dec_dict[spk_id]=cur_Dec
else:
    Dec = model.Decoder(style_dim=len(spk_list), latent_dim=latent_dim, vae_type=args.model_type)
    if args.epoch == 0:
        Dec.load_state_dict(torch.load(model_dir+"/final_dec.pt"))
    else:
        Dec.load_state_dict(torch.load(model_dir+"/parm/"+str(args.epoch)+"_dec.pt"))
    Dec.cuda()
    Dec.eval()

feat_dir = os.path.join("data_seoulNarrative",mode)
#sampling_rate = 22050
sampling_rate = 16000
num_mcep = 36
frame_period = 5.0
n_frames = 128



STAT_DICT = dict()
for source_spk in spk_list:
    stat_path = "data_seoulNarrative/train/"+source_spk+"/feats.p"
    _, sp_m, sp_s, logf0_m, logf0_s = load_pickle(stat_path)
    STAT_DICT[source_spk] = (sp_m, sp_s, logf0_m, logf0_s)
    for target_spk in spk_list:
        os.makedirs(os.path.join(convert_path, source_spk+"_to_"+target_spk), exist_ok=True)

for source_spk in spk_list:
    print("Processing", source_spk)
    feat_path = os.path.join(feat_dir,source_spk)    
    sp_m_s, sp_s_s, logf0_m_s, logf0_s_s = STAT_DICT[source_spk]

    # one hot src
    x = VEC_DICT[source_spk]
    one_hot_x = torch.Tensor(VEC_DICT[source_spk]).cuda()

    x = torch.Tensor(x).float().cuda().contiguous()
    one_hot_x  = torch.reshape(one_hot_x ,(1,TOTAL_SPK_NUM,1))

    for _, _, file_list in os.walk(feat_path):
        for file_id in file_list:
            utt_id = file_id.split(".")[0]
            if utt_id == "ppg36" or utt_id=='feats':
                continue
            print("\tConvert {}.wav ...".format(utt_id))
            file_path = os.path.join(feat_path, file_id)
            sp, src, f0, ap = load_pickle(file_path)

            # src = (coded_sp-sp_m_s) / sp_s_s
            src = np.expand_dims(src, axis=0)
            src = np.expand_dims(src, axis=0)
            src = torch.Tensor(src).float().cuda().contiguous()


            logf0_norm = (np.log(f0)-logf0_m_s) / logf0_s_s

            for target_spk in spk_list:
                style = VEC_DICT[target_spk]
                y = torch.Tensor(style).float().cuda().contiguous()
                one_hot_y = torch.Tensor(style[0]).cuda()
                one_hot_y = torch.reshape(one_hot_y,(1,TOTAL_SPK_NUM,1))

                with torch.no_grad():
                    z_mu, z_logvar, z = Enc(src, y, one_hot_x)
                    if args.model_type == "MD":
                        new_sp, _, _ = Dec_dict[target_spk](z, y)
                    else:
                        new_sp, _, _ = Dec(z, y, one_hot_y)

            
                new_sp = new_sp.double().cpu().numpy()[0][0]
                
                sp_m_t, sp_s_t, logf0_m_t, logf0_s_t = STAT_DICT[target_spk]

                new_f0 = np.exp(logf0_norm * logf0_s_t + logf0_m_t)
                new_sp = new_sp * sp_s_t + sp_m_t
                new_sp = np.ascontiguousarray(new_sp.T)
                new_sp = world_decode_mc(new_sp, fs=sampling_rate)
                
                wav_transformed = world_speech_synthesis(f0=new_f0, decoded_sp=new_sp, 
                    ap=ap, fs=sampling_rate, frame_period=frame_period)
                wav_transformed = np.nan_to_num(wav_transformed)
                soundfile.write(os.path.join(convert_path, source_spk+"_to_"+target_spk, utt_id+".wav"), wav_transformed, sampling_rate)
                
            
            # print(coded_sp.shape)
            # print(f0.shape)
            # print(ap.shape)
