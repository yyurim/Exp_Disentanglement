
python3 -u train_base_ws.py --model_type=VAE3 --seed=0 --model_dir=model_1119/VAE3 --lr_sch=linear15 --conf=conf/rec5kl1.json --epochs=2000 || exit 1;