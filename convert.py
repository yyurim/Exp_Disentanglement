import os, sys
import model
import argparse
import soundfile
import torch
import numpy as np
import pickle

from speech_tools import world_decode_mc, world_speech_synthesis
import data_manager as dm


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

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_normalized_t = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_normalized_t

spk_list = ["VCC2SF1", "VCC2SF2", "VCC2SM1", "VCC2SM2"]
# spk_list = ['F1','M1','F2','M2']
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
parser.add_argument('--disentanglement',type=str,default='')
parser.add_argument('--ws',type=int,default=1)
args = parser.parse_args()

mode = args.mode
assert mode in ['train4eval','test']

model_dir = args.model_path
if args.convert_path=='':
    convert_path = "result/"+args.model_path.split("/")[-1]
else:
    convert_path = args.convert_path
    
os.makedirs(convert_path+"/mean", exist_ok=True)
os.makedirs(convert_path+"/diff", exist_ok=True)
os.makedirs(convert_path+"/sample", exist_ok=True)

latent_dim=8

VAE = model.VAE(style_dim=TOTAL_SPK_NUM, latent_dim=latent_dim, vae_type=args.model_type, weight_sharing=args.ws)
if args.epoch == 0 :
    if args.disentanglement=='':
        args.disentanglement = 'base'
    VAE.load_state_dict(torch.load(model_dir+"/final_{}.pt".format(args.disentanglement)))
else:
    if args.disentanglement=='':
        args.disentanglement = 'base'
    VAE.load_state_dict(torch.load(model_dir+"/parm/"+str(args.epoch)+"_{}.pt".format(args.disentanglement)))
VAE.cuda()
VAE.eval()


# feat_dir = os.path.join("data_seoulNarrative",mode)
feat_dir = os.path.join("data",mode)
sampling_rate = 22050
# sampling_rate = 16000
num_mcep = 36
frame_period = 5.0
n_frames = 128



STAT_DICT = dict()
for source_spk in spk_list:
    stat_path = "data/train/"+source_spk+"/feats.p"
    # stat_path = "data_seoulNarrative/train/"+source_spk+"/feats.p"
    _, sp_m, sp_s, logf0_m, logf0_s = load_pickle(stat_path)
    STAT_DICT[source_spk] = (sp_m, sp_s, logf0_m, logf0_s)
    for target_spk in spk_list:
        os.makedirs(os.path.join(convert_path, 'mean',source_spk+"_to_"+target_spk), exist_ok=True)
        os.makedirs(os.path.join(convert_path, 'diff',source_spk+"_to_"+target_spk), exist_ok=True)
        os.makedirs(os.path.join(convert_path, 'sample',source_spk+"_to_"+target_spk), exist_ok=True)

for (src_idx, source_spk) in SPK_DICT.items():
    print("Processing", source_spk)
    feat_path = os.path.join(feat_dir,source_spk)    
    sp_m_s, sp_s_s, logf0_m_s, logf0_s_s = STAT_DICT[source_spk]

    # one hot src
    # SPK_DICT[source_spk]
    one_hot_x = dm.make_spk_vector([src_idx], TOTAL_SPK_NUM, 1)


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
            
            # logf0_norm = (np.log(f0)-logf0_m_s) / logf0_s_s

            for (tar_idx, target_spk) in SPK_DICT.items():
                # one_hot_y = VEC_DICT[target_spk]
                one_hot_y = dm.make_spk_vector([tar_idx], TOTAL_SPK_NUM, 1)

                with torch.no_grad():
                    _, _,  _, y_prime_mu, _, y_prime = VAE(x=src, one_hot_src=one_hot_x , one_hot_tar=one_hot_y)

                sp_m_t, sp_s_t, logf0_m_t, logf0_s_t = STAT_DICT[target_spk]

                # mean as sample
                mc_t_mean = y_prime_mu.double().cpu().numpy()[0][0]
                mc_t_mean = mc_t_mean * sp_s_t + sp_m_t
                mc_t_mean = mc_t_mean.T
                mc_t_mean = np.ascontiguousarray(mc_t_mean)
                sp_t_mean = world_decode_mc(mc = mc_t_mean, fs = sampling_rate)
                new_sp_mean = sp_t_mean

                # # diff as sample
                # mc_s_mean = x_prime_mu.double().cpu().numpy()[0][0]
                # mc_s_mean = mc_s_mean * sp_s_s + sp_m_s
                # mc_s_mean = mc_s_mean.T
                # mc_s_mean = np.ascontiguousarray(mc_s_mean)
                # sp_s_mean = world_decode_mc(mc = mc_s_mean, fs = sampling_rate)

                # new_sp_diff = sp - sp_s_mean + sp_t_mean
                

                # # sample as sample
                # mc_t_sample = y_prime.double().cpu().numpy()[0][0]
                # mc_t_sample = mc_t_sample * sp_s_t + sp_m_t
                # mc_t_sample = mc_t_sample.T
                # mc_t_sample = np.ascontiguousarray(mc_t_sample)
                # sp_t_sample = world_decode_mc(mc = mc_t_sample, fs = sampling_rate)

                # new_sp_sample = sp_t_sample
                
                # # 여기!!!!
                new_f0 = pitch_conversion(f0 = f0, mean_log_src = logf0_m_s, std_log_src = logf0_s_s, mean_log_target = logf0_m_t, std_log_target = logf0_s_t)

                # mean as sample
                wav_transformed_mean = world_speech_synthesis(f0=new_f0, decoded_sp=new_sp_mean, 
                    ap=ap, fs=sampling_rate, frame_period=frame_period)
                wav_transformed_mean = np.nan_to_num(wav_transformed_mean)
                soundfile.write(os.path.join(convert_path,'mean' ,source_spk+"_to_"+target_spk, utt_id+".wav"), wav_transformed_mean, sampling_rate)

                # # diff as sample
                # wav_transformed_diff = world_speech_synthesis(f0=new_f0, decoded_sp=new_sp_diff, 
                #     ap=ap, fs=sampling_rate, frame_period=frame_period)
                # wav_transformed_diff = np.nan_to_num(wav_transformed_diff)
                # soundfile.write(os.path.join(convert_path,'diff' ,source_spk+"_to_"+target_spk, utt_id+".wav"), wav_transformed_diff, sampling_rate)

                # # sample as sample
                # wav_transformed_sample = world_speech_synthesis(f0=new_f0, decoded_sp=new_sp_sample, 
                #     ap=ap, fs=sampling_rate, frame_period=frame_period)
                # wav_transformed_sample = np.nan_to_num(wav_transformed_sample)
                # soundfile.write(os.path.join(convert_path,'sample' ,source_spk+"_to_"+target_spk, utt_id+".wav"), wav_transformed_sample, sampling_rate)
                
            
            # print(coded_sp.shape)
            # print(f0.shape)
            # print(ap.shape)