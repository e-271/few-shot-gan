c=$1
n=$2
a=0
lr=0.0002
kimg=10000
rr='./pickles/imagenet_10M'
rri=0
rdir='./results' # changeme
dset='celeba'
ddir='./datasets'

python run_training.py --num-gpus=2 --data-dir=$ddir --config=config-$c --dataset=$dset --total-kimg=$kimg --max-images=$n --resume-pkl=$rr --resume-kimg=$rri --rho=$a --lrate-base=$lr --result-dir=$rdir

