mkdir -p stats_1211
mkdir -p stats_1211/VAE3_2AC

echo -e ".....................   Calculate Total loss | Reconstruction loss | MCD | MSD of Test set   .....................\n"
for ep in `seq 0 2000` ; do
    echo '================== Processing epoch '${ep}' ================== '
    python3 -u convert.py --mode=test --model_type=VAE3 --model_path=model_1211/VAE3_2AC --convert_path=result_1211/VAE3_2AC/VAE3_2AC_${ep} --epoch=${ep}  || exit 1;
    python3 -u calculate/calculate.py --gt_dir=corpus_seoulNarrative/test --test_dir=result_1211/VAE3_2AC/VAE3_2AC_${ep} > stats_1211/VAE3_2AC/VAE3_2AC_${ep}.txt || exit 1;
    echo -e '================== Done epoch '${ep}' ================== \n'
done
