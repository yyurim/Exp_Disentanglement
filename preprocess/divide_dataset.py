import glob
import os
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
intersection = list(intersection)[:40]

for spk in train_set:
    for f in train_set[spk]:
        f_name = f.split('/')[-1]
        if f_name in intersection:
            if os.path.isfile(f):
                shutil.move(f, os.path.join(dev_dir,spk,f_name))
            else:
                print("already moved >> ", f)
