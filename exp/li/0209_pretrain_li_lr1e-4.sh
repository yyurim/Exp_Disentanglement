
python3 pretrain_li.py --model_type=VAE3 --ws=0  --model_dir=pretrainLI_0209_lr1e-4 --epochs=2000 --baseline=model_0209/base_nws/parm/4000_base.pt --disentanglement=LI --c_lr=1e-4 || exit;