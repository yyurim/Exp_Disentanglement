
echo -e "TEST"
for ep in `seq 0 2000`; do
    python3 -u test_base.py --model_type=VAE3 --model_path=model_1119/VAE3 --epoch=${ep} --conf=conf_seoulNarr/rec5kl1.json --seed=0 || exit 1;
done
echo -e "\n"
