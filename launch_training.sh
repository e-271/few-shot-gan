#!/bin/bash

# Usage
# ./launch_training.sh N kimg model loss tx eval dir pt idx

if (( $# < 7 )); then
    echo "Usage: ./launch_training.sh N kimg model gloss dloss tx eval dir kw_num gpu dbg"
    exit 1
fi

N=$1
kimg=$2
model=$3 # r, t, s, a, m
lr=0.002
loss=$4 #'G_logistic_ns_gsreg'
dloss=$5
tx=$6
ev=$7
dir=$8
kwn=$9
gpu=${10}
dbg=${11}
#pt=$9
#i=$10


echo $1 $2 $3 $4 $5 $6 $7 $8 #$9 $10
echo $(hostname)
echo $CUDA_VISIBLE_DEVICES
source activate py3tf115


kw=""
if [[ $loss == "pr" ]];
then
loss='G_logistic_ns_pathreg'
elif [[ $loss == "gs" ]];
then
loss='G_logistic_ns_gsreg'
kw={\"gs_weight\":$kwn}

elif [[ $loss == "jc" ]];
then
loss='G_logistic_ns_pathreg_jc'
if [[ $kwn == 0 ]]; then kw='{"epsilon":0.01,"lambda_min":1.5,"lambda_max":1.7}'
elif [[ $kwn == 1 ]]; then kw='{"epsilon":0.1,"lambda_min":1.5,"lambda_max":1.7}'
elif [[ $kwn == 2 ]]; then  kw='{"epsilon":0.01,"lambda_min":1.55,"lambda_max":1.65}'
elif [[ $kwn == 3 ]]; then  kw='{"epsilon":0.01,"lambda_min":1.2,"lambda_max":2.0}'
fi

elif [[ $loss == "div" ]];
then
loss='G_logistic_ns_pathreg_div'
kw={\"div_weight\":$kwn}

elif [[ $loss == "ae" ]];
then
loss="G_logistic_ns_pathreg_ae"
if [[ $kwn == 0 ]]; then kw='{"ae_loss_weight":0.0}'
elif [[ $kwn == 1 ]]; then kw='{"ae_loss_weight":1.0}'
elif [[ $kwn == 2 ]]; then kw='{"ae_loss_weight":0.1}'
fi

fi

if [[ $dloss == "cos" ]];
then
dloss="D_logistic_r1_cos"
else
dloss="D_logistic_r1"

fi


echo $loss

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
if [[ $tx == *"KannadaHnd"* ]]
then
pt='/work/newriver/erobb/pickles/EngFnt.pkl'
elif [[ $tx == *"EngHnd"* ]]
then
pt='/work/newriver/erobb/pickles/EngFnt.pkl'
elif [[ $tx == *"mnist"* ]]
then
pt='/work/newriver/erobb/pickles/EngFnt.pkl'
elif [[ $tx == *"EngFnt"* ]]
then
pt='/work/newriver/erobb/pickles/EngFnt.pkl' # TODO
elif [[ $tx == *"tower"* ]]
then
pt='/work/newriver/erobb/pickles/church-config-f.pkl'
elif [[ $tx == *"bus"* ]]
then
pt='/work/newriver/erobb/pickles/car-config-f.pkl'
elif [[ $tx == *"dog"* ]]
then
pt='/work/newriver/erobb/pickles/cat-config-f.pkl'
elif [[ $tx == *"danbooru"* ]] || [[ $tx == *"anime"* ]] || [[ $tx == *"rei"* ]] || [[ $tx == *"obama"* ]]
then
pt='/work/newriver/erobb/pickles/ffhq-config-f.pkl'
elif [[ $tx == *"cifar10" ]]
then
pt='/work/newriver/erobb/pickles/cifar100_cond.pkl'
elif [[ $tx == *"cifar100" ]]
then
pt='/work/newriver/erobb/pickles/cifar10_cond.pkl'
fi




fi

echo $i $pt


sv=0
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

elif [[ $model == "sv" ]]
then
lr=0.003
cfg="config-sv"
pt=$(echo $pt | sed 's/\.pkl/_svd\.pkl/1')
sv=0

elif [[ $model == "svp" ]]
then
lr=0.003
cfg="config-sv-pkl"
sv=0

fi


if (( $dbg )); then
metrics="cas1k"
else
metrics="fid10k,ppgs10k,lpips10k,cas10k"
fi




echo "CUDA_VISIBLE_DEVICES=$gpu \
python run_training.py \
--num-gpus=1 \
--data-dir=$ddir \
--config=$cfg \
--g-loss=$loss \
--g-loss-kwargs=$kw \
--d-loss=$dloss \
--dataset-train=$tx \
--dataset-eval=$ev \
--total-kimg=$kimg \
--max-images=$N \
--resume-pkl=$pt \
--resume-kimg=$i \
--lrate-base=$lr \
--result-dir=$rdir/$(basename $tx) \
--sv-factors=$sv \
--metrics=$(metrics)"


CUDA_VISIBLE_DEVICES=$gpu \
python run_training.py \
--num-gpus=1 \
--data-dir=$ddir \
--config=$cfg \
--g-loss=$loss \
--g-loss-kwargs=$kw \
--d-loss=$dloss \
--dataset-train=$tx \
--dataset-eval=$ev \
--total-kimg=$kimg \
--max-images=$N \
--resume-pkl=$pt \
--resume-kimg=$i \
--lrate-base=$lr \
--result-dir=$rdir/$(basename $tx) \
--sv-factors=$sv \
--metrics=$metrics

echo "done."
