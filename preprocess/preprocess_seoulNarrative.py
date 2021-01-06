import os
import time

from preprocess_tools import *
import librosa

import glob
import shutil


train_dir = 'corpus_seoulNarrative/train/'
dev_dir = 'corpus_seoulNarrative/dev/'
os.makedirs('corpus_seoulNarrative/dev',exist_ok=True)

spk_list = ['F1','M1','F2','M2']

for spk in spk_list:
    os.makedirs(os.path.join(dev_dir,spk),exist_ok=True)

train_f1 = glob.glob('corpus_seoulNarrative/train/F1/*.wav')
train_m1 = glob.glob('corpus_seoulNarrative/train/M1/*.wav')
train_f2 = glob.glob('corpus_seoulNarrative/train/F2/*.wav')
train_m2 = glob.glob('corpus_seoulNarrative/train/M2/*.wav')

train_set = {'F1':train_f1,'M1':train_m1,'F2':train_f2,'M2':train_m2}

train_m1_set = set([f.split('/')[-1] for f in train_m1])
train_m2_set = set([f.split('/')[-1] for f in train_m2])
train_f1_set = set([f.split('/')[-1] for f in train_f1])
train_f2_set = set([f.split('/')[-1] for f in train_f2])

intersection = train_m1_set&train_m2_set&train_f1_set&train_f2_set
dev_list = list(intersection)[:40]

print(dev_list)

print(len(glob.glob('corpus_seoulNarrative/train/F1/*.wav')))

exp_dir = os.path.join('processed')
start_time = time.time()

#speaker_list = ['F1', 'M1']

sampling_rate = 16000
num_mcep = 36
frame_period = 5.0
n_frames = 128

# train/VCC2SF1 =

dataset_loc = 'data_seoulNarrative'

for speaker in spk_list:
    train_sps = []
    train_f0s = []
    for dtype in ["train", "test"]:
        corpus_dir = os.path.join('corpus_seoulNarrative',dtype,speaker)
        data_dir  = os.path.join(dataset_loc,dtype,speaker)

        non_train_dict = dict()

        # train, dev: feats.p, ppgs.p
        # test: feats.p

        os.makedirs(data_dir, exist_ok=True)
        if dtype == "train":
            dev_dir = os.path.join(dataset_loc, 'dev', speaker)
            os.makedirs(dev_dir, exist_ok=True)

        print('Loading {} Wavs...'.format(speaker))
        f = glob.glob(corpus_dir+'/*')

        for path in f:
            # path = path[:-1]
            info = path.split(".")[-2]
            utt_id = info.split("/")[-1]

            script_id = '_'.join(utt_id.split('_')[1:])

            print("Processing",utt_id)

            wav, _ = librosa.load(path, sr = sampling_rate, mono = True)
            f0, timeaxis, sp, ap, coded_sp = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period, num_mcep=num_mcep)
            frame_num = 4 * (len(f0) // 4)
            coded_sp = coded_sp[:frame_num]
            f0 = f0[:frame_num]
            ap = ap[:frame_num]
            sp = sp[:frame_num]

            if dtype=="train" and utt_id+'.wav' not in dev_list:
                    # train
                    train_f0s.append(f0)
                    train_sps.append(coded_sp.T)
            else:
                print("     is {} >> ".format('dev' if dtype=='train' else dtype),utt_id)
                non_train_dict[utt_id] = (sp, coded_sp.T, f0, ap)

        # os.system("rm flist.txt")

        print('Saving {} data to {}... '.format(speaker, dtype))
        if dtype=='train':
            log_f0s_mean, log_f0s_std = logf0_statistics(train_f0s)
            train_sps_norm, sps_mean, sps_std = mcs_normalization_fit_transform(mcs=train_sps)
            save_pickle(os.path.join(dataset_loc, 'train', speaker, 'feats.p'),
                        (train_sps_norm, sps_mean, sps_std, log_f0s_mean, log_f0s_std))

            data_dir = os.path.join(dataset_loc, 'dev', speaker)
            corpus_dir_dev = os.path.join('corpus_seoulNarrative','dev',speaker)
            
            for utt_id, (sp , coded_sp, f0, ap) in non_train_dict.items():
                # print("Processing dev >> ",utt_id)
                new_sp = (coded_sp-sps_mean) / sps_std
                save_pickle(os.path.join(data_dir, '{}.p'.format(utt_id)), (new_sp, f0, ap))

                shutil.move(os.path.join(corpus_dir,utt_id+'.wav'), os.path.join(corpus_dir_dev,utt_id+'.wav'))
                print("     moved dev {}.wav ? >> ".format(utt_id), os.path.isfile(os.path.join(corpus_dir_dev,utt_id+'.wav')))
        else:
            data_dir = os.path.join(dataset_loc, dtype, speaker)
            for utt_id, (sp , coded_sp, f0, ap) in non_train_dict.items():
                print("Processing",utt_id)
                new_sp = (coded_sp-sps_mean) / sps_std
                save_pickle(os.path.join(data_dir, '{}.p'.format(utt_id)), (sp, new_sp, f0, ap))

        
    print('Preprocessing Done.')
