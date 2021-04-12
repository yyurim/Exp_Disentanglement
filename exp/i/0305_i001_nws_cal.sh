mkdir -p stats_0305

mkdir -p stats_0305/i001

echo -e ".....................   Calculate Total loss | Reconstruction loss | MCD | MSD of Test set   .....................\n"
# for ep in 1900  ; do
for ep in `seq 3999 -2 2000` ; do
    echo '================== Processing epoch '${ep}' ================== '
    python3 -u convert.py --mode=test --model_type=VAE3 --model_path=model_0305/i001 --convert_path=result_0305/i001/i001_${ep} --epoch=${ep} --disentanglement=I --ws=0 || exit 1;
    python3 -u calculate/calculate_exp.py --gt_dir=corpus/test --test_dir=result_0305/i001/i001_${ep} > stats_0305/i001/i001_${ep}.txt || exit 1;

    rm -rf result_0305/i001/i001_${ep}/*/*.wav
    rm -rf result_0305/i001/i001_${ep}/*
    rm -rf result_0305/i001/i001_${ep}
    echo -e '================== Done epoch '${ep}' ================== \n'
done
