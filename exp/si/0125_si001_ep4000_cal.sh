mkdir -p stats_0125

mkdir -p stats_0125/si001_ep4000

echo -e ".....................   Calculate Total loss | Reconstruction loss | MCD | MSD of Test set   .....................\n"
# for ep in 1900  ; do
for ep in `seq 3999 -2 2000` ; do
    echo '================== Processing epoch '${ep}' ================== '
    python3 -u convert.py --mode=test --model_type=VAE3 --model_path=model_0125/si001_ep4000 --convert_path=result_0125/si001_ep4000/si001_ep4000_${ep} --epoch=${ep} --disentanglement=SI|| exit 1;
    python3 -u calculate/calculate_exp.py --gt_dir=corpus/test --test_dir=result_0125/si001_ep4000/si001_ep4000_${ep} > stats_0125/si001_ep4000/si001_ep4000_${ep}.txt || exit 1;

    rm -rf result_0125/si001_ep4000/si001_ep4000_${ep}/*/*.wav
    rm -rf result_0125/si001_ep4000/si001_ep4000_${ep}/*
    rm -rf result_0125/si001_ep4000/si001_ep4000_${ep}
    echo -e '================== Done epoch '${ep}' ================== \n'
done
