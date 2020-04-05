#!/bin/bash

echo $(hostname)

if [[ $(hostname) == "ca"* ]];
then


dir="pretrains"
N=100000
kimg=25000
m=v
loss="pr"

#for tx in "mnist" "font"
#do
#qsub sgan_ca.qsub $N $kimg $m $loss $tx $tx $dir
#done

dir="loss_testing"
N=25
kimg=1000
m=t
tx="tower25"
ev="tower"

loss=div
echo "qsub sgan_ca.qsub $N $kimg $m $loss $tx $ev $dir"
qsub sgan_ca.qsub $N $kimg $m $loss $tx $ev $dir

#loss=gs
#kw='{"epsilon":0.01}'
#qsub sgan_ca.qsub $N $kimg $m $loss $tx $ev $dir $kw



elif [[ $(hostname) == "nr"* ]];
then

dir="personalization"
N=25
kimg=100
for tx in "obama25" "rei12"
do
for m in r
do
qsub sgan_nr.qsub -v "N=$N, model=$m, tx=$tx, kimg=$kimg, dir=$dir"
done
done

qsub sgan_nr.qsub -v "N=$N, model=t, tx=$tx, kimg=$kimg, dir=$dir"




# Run on some local machine
else
lr=0.0002
rho=0
tx="tower25"
kimg=100

CUDA_VISIBLE_DEVICES=2,4 nohup ./launch_training.sh 25 t $rho $lr $tx $kimg > ${tx}_t_25.out 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 nohup ./launch_training.sh 25 r $rho $lr $tx $kimg > ${tx}_r_25.out 2>&1 &
#tx='kannada4K'
#CUDA_VISIBLE_DEVICES=8,9 nohup ./launch_training.sh 10 t $rho $lr $tx 10000 > ${tx}_t_10.out 2>&1 &
#nohup ./launch_training.sh 50 t $rho $lr $tx > ${tx}_t_50.out 2>&1 &
#nohup ./launch_training.sh 100 r $rho $lr $tx > ${tx}_r_100.out 2>&1 &

fi
