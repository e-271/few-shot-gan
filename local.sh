#!/bin/bash
kimg=1000
rr='pickles/imagenet_5M.pkl'
rri=0
rdir='./results/celeba/' # changeme
dset='celeba'
ddir='./datasets/'

python run_training.py --num-gpus=2 --data-dir=$ddir --config=config-$c --dataset=$dset --total-kimg=$kimg --max-images=$n --resume-pkl=$rr --resume-kimg=$rri --rho=$a --lrate-base=$lr --result-dir=$rdir

