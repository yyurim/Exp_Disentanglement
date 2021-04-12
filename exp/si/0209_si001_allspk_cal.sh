mkdir -p stats_0209

mkdir -p stats_0209/si001_allspk

echo -e ".....................   Calculate Total loss | Reconstruction loss | MCD | MSD of Test set   .....................\n"
# for ep in 1900  ; do
for ep in `seq 3999 -2 2000` ; do
    echo '================== Processing epoch '${ep}' ================== '
    python3 -u convert.py --mode=test --model_type=VAE3 --model_path=model_0206/si001_ep4000 --convert_path=result_0209/si001_allspk/si001_allspk_${ep} --epoch=${ep} --spk=all --disentanglement=SI || exit 1;
    python3 -u calculate/calculate_exp.py --gt_dir=corpus/test --test_dir=result_0209/si001_allspk/si001_allspk_${ep} > stats_0209/si001_allspk/si001_allspk_${ep}.txt || exit 1;

    rm -rf result_0209/si001_allspk/si001_allspk_${ep}/*/*.wav
    rm -rf result_0209/si001_allspk/si001_allspk_${ep}/*
    rm -rf result_0209/si001_allspk/si001_allspk_${ep}
    echo -e '================== Done epoch '${ep}' ================== \n'
done
