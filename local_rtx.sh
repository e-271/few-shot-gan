c=$1
n=$2
a=0
lr=0.0002
kimg=10000
rr='/work/erobb/results/00003-stylegan2-imagenet_train-8gpu-config-f/network-snapshot-008908.pkl'
rri=8908
rdir='/work/erobb/results/' # changeme
dset='imagenet_train'
ddir='/work/erobb/'

CUDA_VISBLE_DEVICES=0,1,2,3,4,5,6,7 python run_training.py --num-gpus=8 --data-dir=$ddir --config=config-$c --dataset=$dset --total-kimg=$kimg --max-images=$n --resume-pkl=$rr --resume-kimg=$rri --rho=$a --lrate-base=$lr --result-dir=$rdir

