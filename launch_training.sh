#!/bin/bash

# Usage
# ./launch_training.py 100 a

N=$1
model=$2 # r, t, s, a
reg=$3
lr=$4
kimg=10000
tx=$5

echo $1 $2 $3 $4 $5

hostname
echo $CUDA_VISIBLE_DEVICES
source activate py3tf115


if [[ $tx == "kannada4K" ]]
then

rv=''
rvi=0
rt='./pickles/eng-config-f-10M.pkl'
rti=0
rs='./pickles/eng-config-f-10M.pkl'
rsi=0
rr='./pickles/eng-config-f-10M.pkl'
rri=0

# Resume from pretrained checkpoint
if [[ $N == 10 ]];
then
rt='results/00046-stylegan2-kannada4K-2gpu-config-f-10img-rho0.0E00/network-snapshot-003014.pkl'
rti=3014
rr='results/00061-stylegan2-kannada4K-2gpu-config-c-b-10img-rho0.0E00-lr2.0E-04/network-snapshot-004019.pkl'
rri=4019

elif [[ $N == 50 ]];
then
rr='results/00064-stylegan2-kannada4K-2gpu-config-c-b-50img-rho0.0E00-lr2.0E-04/network-snapshot-004019.pkl'
rri=4019
rt='results/00068-stylegan2-kannada4K-2gpu-config-f-50img-rho0.0E00-lr2.0E-04/network-snapshot-004019.pkl'
rti=4019

elif [[ $N == 100 ]];
then
rr='results/00062-stylegan2-kannada4K-2gpu-config-c-b-100img-rho0.0E00-lr2.0E-04/network-snapshot-006028.pkl'
rri=6028

fi

elif [[ $tx == "danbooru" ]]
then

rv=''
rvi=0
rt='./pickles/imagenet_5M.pkl'
rti=0
rs='./pickles/imagenet_5M.pkl'
rsi=0
rr='./pickles/imagenet_5M.pkl'
rri=0

fi

# Vanilla GAN baseline
if [[ $model == "v" ]]
then
python run_training.py --num-gpus=2 --data-dir=./datasets --config=config-f --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$rv --resume-kimg=$rvi --lrate-base=$lr --result-dir=./results/$tx
echo "vvv"

# TGAN baseline
elif [[ $model == "t" ]]
then
echo $rt $rti
python run_training.py --num-gpus=2 --data-dir=./datasets --config=config-f --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$rt --resume-kimg=$rti --lrate-base=$lr --result-dir=./results/$tx
echo "ttt"

# Shift+scale baseline
elif [[ $model == "s" ]]
then
python run_training.py --num-gpus=2 --data-dir=./datasets --config=config-a-gb --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$rs --resume-kimg=$rsi --rho=$reg --lrate-base=$lr --result-dir=./results/$tx
echo "sss"

# Residual adapters (ours#)
elif [[ $model == "r" ]]
then
python run_training.py --num-gpus=2 --data-dir=./datasets --config=config-c-b --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$rr --resume-kimg=$rri --rho=$reg --lrate-base=$lr --result-dir=./results/$tx
echo "rrr"
fi

