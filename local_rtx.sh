c=$1
n=$2
a=0
lr=0.0002
kimg=1000
rr='pickles/imagenet_5M.pkl'
rri=0
rdir='/work/erobb/results/' # changeme
dset='celeba'
ddir='/work/erobb/datasets/'

CUDA_VISBLE_DEVICES=8 python run_training.py --num-gpus=1 --data-dir=$ddir --config=config-$c --dataset=$dset --total-kimg=$kimg --max-images=$n --resume-pkl=$rr --resume-kimg=$rri --rho=$a --lrate-base=$lr --result-dir=$rdir

