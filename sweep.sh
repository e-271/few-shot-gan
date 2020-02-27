#!/bin/bash


if [[ $(hostname) == *"ca"* ]];
then

lr=0.0002
a=0
tx="kannada4K"

qsub sgan_ca.qsub 10 t $a $lr $tx
#qsub sgan_ca.qsub 10 r $a $lr $tx
#qsub sgan_ca.qsub 50 r $a $lr $tx
#qsub sgan_ca.qsub 100 r $a $lr $tx


#for n in 1 10 100 1000 4000
#do
#for a in 0.1 0.01 0.001
#do
#for lr in 0.00002
#do
#qsub sgan_ca.qsub $n r $a $lr
#done
#done
#done


else
lr=0.0002
a=0
tx="kannada4K"


qsub sgan_nr.qsub -v "N=10, model=t, rho=$a, lrate=$lr, tx=$tx"
qsub sgan_nr.qsub -v "N=10, model=r, rho=$a, lrate=$lr, tx=$tx"
qsub sgan_nr.qsub -v "N=50, model=t, rho=$a, lrate=$lr, tx=$tx"
qsub sgan_nr.qsub -v "N=100, model=r, rho=$a, lrate=$lr, tx=$tx"




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

fi
