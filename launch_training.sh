#!/bin/bash

# Usage
# ./launch_training.sh N kimg model loss tx eval dir pt idx

if (( $# < 7 )); then
    echo "Usage: ./launch_training.sh N kimg model loss tx eval dir (pt) (idx)"
    exit 1
fi

N=$1
kimg=$2
model=$3 # r, t, s, a
lr=0.002
loss=$4 #'G_logistic_ns_gsreg'
tx=$5
ev=$6
dir=$7
pt=$8
i=$9


echo $1 $2 $3 $4 $5 $6 $7 $8
echo $(hostname)
echo $CUDA_VISIBLE_DEVICES
source activate py3tf115


if [[ $loss == "pr" ]];
then
loss='G_logistic_ns_pathreg'

elif [[ $loss == "gs" ]];
then
loss='G_logistic_ns_gsreg'

elif [[ $loss == "jc" ]];
then
loss='G_logistic_ns_pathreg_jc'
fi
echo $loss $loss



if [[ $(hostname) == "jb"* ]]; # RTX
then
ddir='/work/erobb/datasets/'
rdir="/work/erobb/results/$dir"
else # ARC
ddir='/work/newriver/erobb/datasets'
rdir="/work/newriver/erobb/results/$dir"
fi

if [[ $pt == "" ]]
then
i=0
echo "automatically choosing pretrain"
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

echo $i $pt


# Vanilla GAN baseline
if [[ $model == "v" ]]
then
cfg="config-f"
pt=""
# TGAN baseline
elif [[ $model == "t" ]]
then
cfg="config-f"
# Shift+scale baseline
elif [[ $model == "s" ]]
then
cfg="config-ss"
# Residual adapters
elif [[ $model == "r" ]]
then
cfg="config-ra"
fi


python run_training.py \
--num-gpus=2 \
--data-dir=$ddir \
--config=$cfg \
--g-loss=$loss \
--dataset-train=$tx \
--dataset-eval=$ev \
--total-kimg=$kimg \
--max-images=$N \
--resume-pkl=$pt \
--resume-kimg=$i \
--lrate-base=$lr \
--result-dir=$rdir/$tx

echo "done."
