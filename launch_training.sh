#!/bin/bash

# Usage
# ./launch_training.py N model tx kimg dir pt idx

N=$1
model=$2 # r, t, s, a
reg=0
lr=0.002
tx=$3
kimg=$4
dir=$5
pt=$6
i=$7


if [[ $(hostname) == "jb"* ]]; # RTX
then

ddir='/work/erobb/datasets/'
rdir="/work/erobb/results/$7"

else # CA or NR

ddir='./datasets'
rdir="/work/newriver/erobb/results/$7"

fi


echo $1 $2 $3 $4 $5 $6 $7

hostname
echo $CUDA_VISIBLE_DEVICES
source activate py3tf115


echo $pt

if [[ $pt == "" ]]
then

echo "automatically choosing pretrain"
i=0

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


elif [[ $tx == "danbooru"* ]] || [[ $tx == "anime"* ]] || [[ $tx == "rei"* ]] || [[ $tx == "obama"* ]]
then
pt='./pickles/ffhq-config-f.pkl'

fi
fi

echo $pt




#rv=''
#rvi=0
#rt=$pt
#rti=0
#rs=$pt
#rsi=0
#rr=$pt
#rri=0

#rt="/work/newriver/erobb/results/table3/dog25/00001-stylegan2-dog25-2gpu-config-f-25img-rho0.0E00-lr2.0E-04/network-snapshot-000143.pkl"
#rti=143
#rr="/work/newriver/erobb/results/table3/anime25/00015-stylegan2-anime25-2gpu-config-c-b-25img-rho0.0E00-lr2.0E-04-aug/network-snapshot-000102.pkl"
#rri=102


# Vanilla GAN baseline
if [[ $model == "v" ]]
then
python run_training.py --num-gpus=2 --data-dir=$ddir --config=config-f --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$pt --resume-kimg=$i --lrate-base=$lr --result-dir=$rdir/$tx

# TGAN baseline
elif [[ $model == "t" ]]
then
python run_training.py --num-gpus=2 --data-dir=$ddir --config=config-f --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$pt --resume-kimg=$i --lrate-base=$lr --result-dir=$rdir/$tx

# Shift+scale baseline
elif [[ $model == "s" ]]
then
python run_training.py --num-gpus=2 --data-dir=$ddir --config=config-a-gb --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$pt --resume-kimg=$i --rho=$reg --lrate-base=$lr --result-dir=$rdir/$tx

# Residual adapters (ours#)
elif [[ $model == "r" ]]
then
python run_training.py --num-gpus=2 --data-dir=$ddir --config=config-c-b --dataset=$tx --total-kimg=$kimg --max-images=$N --resume-pkl=$pt --resume-kimg=$i --rho=$reg --lrate-base=$lr --result-dir=$rdir/$tx
fi

echo "done."
