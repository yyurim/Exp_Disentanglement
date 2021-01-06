import os

from preprocess_tools import *

speaker_list = ['VCC2SF1','VCC2SF2','VCC2SM1','VCC2SM2'] 
dev_list = [10023, 10040, 10013, 10054, 10027, 10056, 10037, 10021, 10033, 
10062, 10026, 10078, 10016, 10034, 10006, 10002, 10046]


sampling_rate = 22050
num_mcep = 36
frame_period = 5.0
n_frames = 128




# train/VCC2SF1 =

for speaker in speaker_list:
    train_sps = []
    train_f0s = []
    for dtype in ["train", "test"]:
        corpus_dir = os.path.join('corpus', dtype, speaker)
        data_dir = os.path.join('data', dtype, speaker)

        non_train_dict = dict()

        # train, dev: feats.p, ppgs.p
        # test: feats.p 

        os.makedirs(data_dir, exist_ok=True)
        if dtype == "train":
            dev_dir = os.path.join('data', 'dev', speaker)
            os.makedirs(dev_dir, exist_ok=True)


        cmd="find "+corpus_dir+" -iname '*.wav' > flist.txt"
        os.system(cmd)
            
        print('Loading {} Wavs...'.format(speaker))
        with open("flist.txt", 'r') as f:
            for path in f:
                path = path[:-1]
                info = path.split(".")[-2]
                utt_id = info.split("/")[-1]
                print("Processing",utt_id)
                wav, _ = librosa.load(path, sr = sampling_rate, mono = True)
                f0, timeaxis, sp, ap, coded_sp = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period, num_mcep=num_mcep)
                frame_num = 4 * (len(f0) // 4)
                coded_sp = coded_sp[:frame_num]
                f0 = f0[:frame_num]
                ap = ap[:frame_num]

                if dtype=="train" and int(utt_id) not in dev_list:
                        # train
                        train_f0s.append(f0)
                        train_sps.append(coded_sp.T)
                else:
                    non_train_dict[utt_id] = (coded_sp.T, f0, ap)

        os.system("rm flist.txt")

        print('Saving {} data to {}... '.format(speaker, dtype))
        if dtype=='train':
            log_f0s_mean, log_f0s_std = logf0_statistics(train_f0s)
            train_sps_norm, sps_mean, sps_std = mcs_normalization_fit_transform(mcs=train_sps)
            save_pickle(os.path.join('data', 'train', speaker, 'feats.p'),
                        (train_sps_norm, sps_mean, sps_std, log_f0s_mean, log_f0s_std))
            data_dir = os.path.join('data', 'dev', speaker)
        else:
            data_dir = os.path.join('data', 'test', speaker)

        for utt_id, (sp, f0, ap) in non_train_dict.items():
            new_sp = (sp-sps_mean) / sps_std
            save_pickle(os.path.join(data_dir, '{}.p'.format(utt_id)), (new_sp, f0, ap))
        
 

    print('Preprocessing Done.')