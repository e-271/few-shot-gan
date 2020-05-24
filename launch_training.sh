#!/bin/bash

# Usage
# ./launch_training.sh N kimg model loss tx eval dir pt idx

if (( $# < 7 )); then
    echo "Usage: ./launch_training.sh N kimg model spc tx eval dir gpu dbg rep"
    exit 1
fi

N=$1
kimg=$2
model=$3 # r, t, s, a, m
spc=$4
tx=$5
ev=$6
dir=$7
gpu=$8
dbg=$9
rep=${10}

lr=0.002
metrics=""
pt=""
aug=0
i=0
nt=1

echo $1 $2 $3 $4 $5 $6 $7 $8
echo $(hostname)
echo $CUDA_VISIBLE_DEVICES


if [[ $(hostname) == "jb"* ]]; # RTX
then
ddir='/work/erobb/datasets/'
rdir="/work/erobb/results/$dir/$(basename $tx)_${N}shot/$model"
else # ARC
ddir='/work/newriver/erobb/datasets'
rdir="/work/newriver/erobb/results/$dir/$(basename $tx)_${N}shot/$model"
fi


if [[ $pt == "" ]]
then

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
aug=1
elif [[ $tx == *"bus"* ]]
then
pt='/work/newriver/erobb/pickles/car-config-f.pkl'
aug=1
elif [[ $tx == *"dog"* ]]
then
pt='/work/newriver/erobb/pickles/cat-config-f.pkl'
aug=1
elif [[ $tx == *"danbooru"* ]] || [[ $tx == *"anime"* ]] || [[ $tx == *"rei"* ]] || [[ $tx == *"obama"* ]]
then
pt='/work/newriver/erobb/pickles/ffhq-config-f.pkl'
aug=1
elif [[ $tx == *"cifar10" ]]
then
pt='/work/newriver/erobb/pickles/cifar100_cond.pkl'
metrics='cas10k,'
aug=1
nt=4
elif [[ $tx == *"cifar100"* ]]
then
pt='/work/newriver/erobb/pickles/cifar10_cond.pkl'
metrics='cas10k,'
aug=1
nt=4
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

elif [[ $model == "svm" ]]
then
lr=0.003
cfg="config-sv-map"
sv=0

elif [[ $model == "svs" ]]
then
lr=0.003
cfg="config-sv-syn"
sv=0

elif [[ $model == "sva" ]]
then
lr=0.003
cfg="config-sv-all"
sv=0

fi




if (( $dbg )); then
metrics="fid1k"
else
metrics="fid10k,ppgs1k"
fi


for i in $(seq 1 $rep)
do
echo "CUDA_VISIBLE_DEVICES=$gpu \
python run_training.py \
--num-gpus=1 \
--data-dir=$ddir \
--config=$cfg \
--dataset-train=$tx \
--dataset-eval=$ev \
--total-kimg=$kimg \
--max-images=$N \
--resume-pkl=$pt \
--resume-kimg=$i \
--lrate-base=$lr \
--result-dir=$rdir \
--sv-factors=$sv \
--mirror-augment=$aug\
--net-ticks=$nt \
--metrics=$metrics \
--skip-images=-$i"


CUDA_VISIBLE_DEVICES=$gpu \
python run_training.py \
--num-gpus=1 \
--data-dir=$ddir \
--config=$cfg \
--spatial-svd=$spc \
--dataset-train=$tx \
--dataset-eval=$ev \
--total-kimg=$kimg \
--max-images=$N \
--resume-pkl=$pt \
--resume-kimg=$i \
--lrate-base=$lr \
--result-dir=$rdir \
--sv-factors=$sv \
--mirror-augment=$aug \
--net-ticks=$nt \
--metrics=$metrics \
--skip-images=-$i
done

echo "done."

