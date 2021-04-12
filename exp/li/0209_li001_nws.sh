
python3 -u train.py --model_type=VAE3 --disentanglement=LI --ws=0 --li_lr_sch=0 --model_dir=model_0209/li001_nws --conf=conf/LI/100LI.json --li_path=pretrainLI_0209/li_2000.pt --epochs=4000 --c_lr=0.05 || exit 1;

