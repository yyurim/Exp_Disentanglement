
python3 -u train.py --model_type=VAE3 --disentanglement=SI --ws=0  --si_lr_sch=0 --si_path=pretrainSI_0209_nws/si_2000.pt --model_dir=model_0209/si01_nws --conf=conf/SI/10SI.json  --epochs=4000 --c_lr=0.05 || exit 1;

