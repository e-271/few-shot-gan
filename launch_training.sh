#!/bin/bash

# Usage
# ./launch_training.sh N kimg model gloss tx eval dir pt idx

if (( $# < 9 )); then
    echo "Usage: ./launch_training.sh N kimg model gloss dloss mi tx eval dir (kw_num) (gpu)"
    exit 1
fi

N=$1
kimg=$2
model=$3 # r, t, s, a, m
lr=0.002
gloss=$4 #'G_logistic_ns_gsreg'
dloss=$5
mi=$6
tx=$7
ev=$8
dir=$9
kwn=${10} 
gpu=${11}
#pt=$9
#i=$10


echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11}
echo $(hostname)
echo $CUDA_VISIBLE_DEVICES
source activate py3tf115


gkw=""
dkw=""
if [[ $gloss == "pr" ]];
then
    gloss='G_logistic_ns_pathreg'
elif [[ $gloss == "gs" ]];
then
    gloss='G_logistic_ns_gsreg'
    if [[ $gkwn == 0 ]]; then gkw='{"gs_weight":5.0}'
    elif [[ $gkwn == 1 ]]; then gkw='{"gs_weight":10.0}'
    elif [[ $gkwn == 2 ]]; then gkw='{"gs_weight":3.0}'
    fi
elif [[ $gloss == "jc" ]];
then
    gloss='G_logistic_ns_pathreg_jc'
    if [[ $gkwn == 0 ]]; then gkw='{"epsilon":0.01,"lambda_min":1.5,"lambda_max":1.7}'
    elif [[ $gkwn == 1 ]]; then gkw='{"epsilon":0.1,"lambda_min":1.5,"lambda_max":1.7}'
    elif [[ $gkwn == 2 ]]; then  gkw='{"epsilon":0.01,"lambda_min":1.55,"lambda_max":1.65}'
    elif [[ $gkwn == 3 ]]; then  gkw='{"epsilon":0.01,"lambda_min":1.2,"lambda_max":2.0}'
    fi
elif [[ $gloss == "div" ]];
then
    gloss='G_logistic_ns_pathreg_div'
fi

if [[ $dloss == "cos" ]];
    then
    dloss="D_logistic_r1_cos"
else
    dloss="D_logistic_r1"
fi



if [[ $mi == "ae" ]];
then
    gloss="G_logistic_ns_pathreg_ae"
    mi="config-ae"
    if [[ $kwn == 0 ]]; then gkw='{"ae_loss_weight":0.0}'
    elif [[ $kwn == 1 ]]; then gkw='{"ae_loss_weight":1.0}'
    elif [[ $kwn == 2 ]]; then gkw='{"ae_loss_weight":0.1}'
    fi
elif [[ $mi == "ft" ]];
    then
    gloss="G_logistic_ns_pathreg_ft"
    dloss="D_logistic_r1_ft"
    mi="config-ft"
    gkw='{"mi_weight":0.1}'
    dkw='{"mi_weight":0.1}'
fi


echo $gloss

if [[ $(hostname) == "jb"* ]]; # RTX
then
ddir='/work/erobb/datasets/'
rdir="/work/erobb/results/$dir"
else # ARC
ddir='/work/newriver/erobb/datasets'
rdir="/work/newriver/erobb/results/$dir"
fi

pt=""
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
lr=0.0003
cfg="config-ss"
# Residual adapters
elif [[ $model == "r" ]]
then
lr=0.0003
cfg="config-ra"
# ReLU Residual adapters
elif [[ $model == "rr" ]]
then
lr=0.0003
cfg="config-rr"
# AdaIN Residual adapters
elif [[ $model == "ar" ]]
then
lr=0.0003
cfg="config-ar"

fi




echo "CUDA_VISIBLE_DEVICES=$gpu \
python run_training.py \
--num-gpus=1 \
--data-dir=$ddir \
--config=$cfg \
--g-loss=$gloss \
--g-loss-kwargs=$gkw \
--d-loss=$dloss \
--d-loss-kwargs=$dkw \
--mi-config=$mi \
--dataset-train=$tx \
--dataset-eval=$ev \
--total-kimg=$kimg \
--max-images=$N \
--resume-pkl=$pt \
--resume-kimg=$i \
--lrate-base=$lr \
--result-dir=$rdir/$tx \
--metrics=fid1k"


echo "done."
