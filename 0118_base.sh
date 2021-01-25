mkdir -p model_0118

python3 -u train.py --model_type=VAE3 --model_dir=model_0118/base --conf=conf/base/51.json --disentanglement=base  || exit 1;