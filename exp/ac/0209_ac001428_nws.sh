
python3 -u train.py --model_type=VAE3 --disentanglement=AC --ws=0  --si_lr_sch=0 --ac_path=pretrainAC_0209_nws/ac_2000.pt --model_dir=model_0209/ac001428_nws --conf=conf/AC/70AC.json  --epochs=4000 --c_lr=0.05 || exit 1;

