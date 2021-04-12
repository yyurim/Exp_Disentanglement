
python3 -u train.py --model_type=VAE3 --disentanglement=LI --ws=0 --li_lr_sch=0 --model_dir=model_0209/li001_nws_lr1e-2 --conf=conf/LI/100LI.json --li_path=pretrainLI_0209_lr1e-2/li_2000.pt --epochs=4000 --c_lr=1e-2 || exit 1;

