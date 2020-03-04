#!/bin/bash


if [[ $(hostname) == "ca"* ]];
then
echo $(hostname)
lr=0.0002
a=0
tx="anime25"
kimg=500
dir="teaser"
for N in 9 #1 10 25
do
for m in r #t s
do

qsub sgan_ca.qsub $N $m $a $lr $tx $kimg $dir

done
done



elif [[ $(hostname) == "nr"* ]];
then
lr=0.0002
a=0
#tx="kannada4K"
#tx="anime25"
tx="tower25"
#tx="dog25"
kimg=200
dir="table1"
#dir="table3"

for N in 1 10 25
do
for m in r t s
do

qsub sgan_nr.qsub -v "N=$N, model=$m, rho=$a, lrate=$lr, tx=$tx, kimg=$kimg, dir=$dir"

done
done


#qsub sgan_nr.qsub -v "N=50, model=r, rho=$a, lrate=$lr, tx=$tx"
#qsub sgan_nr.qsub -v "N=50, model=v, rho=$a, lrate=$lr, tx=$tx"
#qsub sgan_nr.qsub -v "N=50, model=t, rho=$a, lrate=$lr, tx=$tx"


#for n in 50 #10 100 1000 4000
#do
#for a in 0 #0.05 0.005 0.0005
#do
#for lr in 0.0002
#do
#qsub sgan_nr.qsub -v "N=$n, model=r, rho=$a, lrate=$lr"
#done
#done
#done


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
