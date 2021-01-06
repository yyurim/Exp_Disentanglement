import numpy as np
import pyworld
import pickle
import os
import pysptk

def coded_sps_normalization_transform(coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    
    return coded_sps_normalized

def coded_sps_normalization_inverse_transform(normalized_coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps

def coded_sp_padding(coded_sp, multiple = 4):
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values = 0)

    return coded_sp_padded

def world_encode_data(wavs, fs, frame_period = 5.0, num_mcep = 36):

    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    mcs = list()

    for wav in wavs:
        f0, timeaxis, sp, ap, mc = world_decompose(wav = wav, fs = fs, frame_period = frame_period, num_mcep=num_mcep)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        mcs.append(mc)

    return f0s, timeaxes, sps, aps, mcs

def world_decompose(wav, fs, frame_period = 5.0, num_mcep=36):

    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.conversion.sp2mc(sp, order=num_mcep-1, alpha=alpha)

    return f0, timeaxis, sp, ap, mc

def logf0_statistics(f0s):

    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std


def transpose_in_list(lst):

    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst


def mcs_normalization_fit_transform(mcs):

    mcs_concatenated = np.concatenate(mcs, axis = 1)
    mcs_mean = np.mean(mcs_concatenated, axis = 1, keepdims = True)
    mcs_std = np.std(mcs_concatenated, axis = 1, keepdims = True)

    mcs_normalized = list()
    for mc in mcs:
        mcs_normalized.append((mc - mcs_mean) / mcs_std)
    
    return mcs_normalized, mcs_mean, mcs_std
  
def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1 
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

    nlen = len(wav_padded) + 80
    a = 2**5
    wav_padded = np.pad(wav_padded, (0, (a-(nlen//80)%a)*80 - (nlen%80)), 'constant', constant_values = 0)
    
    return wav_padded

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
