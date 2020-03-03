#!/bin/bash

# Usage
# ./launch_training.py 100 a

N=$1
model=$2 # r, t, s, a
reg=$3
lr=$4
tx=$5
kimg=$6
dir=$7

if [[ $(hostname) == "jb"* ]]; # RTX
then

ddir='/work/erobb/datasets/'
rdir='/work/erobb/results/$7'

else

ddir='./datasets'
rdir='./results/$7'

fi


echo $1 $2 $3 $4 $5

hostname
echo $CUDA_VISIBLE_DEVICES
source activate py3tf115


if [[ $tx == "kannada"* ]]
then
pt='./pickles/eng-config-f-10M.pkl'

elif [[ $tx == "tower"* ]]
then
pt='./pickles/church-config-f.pkl'

elif [[ $tx == "bus"* ]]
then
pt='./pickles/car-config-f.pkl'


elif [[ $tx == "dog"* ]]
then
pt='./pickles/cat-config-f.pkl'


elif [[ $tx == "danbooru"* ]] | [[ $tx == "anime"* ]]
then
pt='./pickles/ffhq-config-f.pkl'

fi




rv=''
rvi=0
rt=$pt
rti=0
rs=$pt
rsi=0
rr=$pt
rri=0

# Vanilla GAN baseline
if [[ $model == "v" ]]
then
python run_training.py --num-gpus=2 --data-dir=$ddir --config=config-f --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$rv --resume-kimg=$rvi --lrate-base=$lr --result-dir=$rdir/$tx
echo "vvv"

# TGAN baseline
elif [[ $model == "t" ]]
then
echo $rt $rti
python run_training.py --num-gpus=2 --data-dir=$ddir --config=config-f --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$rt --resume-kimg=$rti --lrate-base=$lr --result-dir=$rdir/$tx
echo "ttt"

# Shift+scale baseline
elif [[ $model == "s" ]]
then
python run_training.py --num-gpus=2 --data-dir=$ddir --config=config-a-gb --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$rs --resume-kimg=$rsi --rho=$reg --lrate-base=$lr --result-dir=$rdir/$tx
echo "sss"

# Residual adapters (ours#)
elif [[ $model == "r" ]]
then
python run_training.py --num-gpus=2 --data-dir=$ddir --config=config-c-b --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$rr --resume-kimg=$rri --rho=$reg --lrate-base=$lr --result-dir=$rdir/$tx
echo "rrr"
fi

