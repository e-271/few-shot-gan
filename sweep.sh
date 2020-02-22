#!/bin/bash


if [[ $(hostname) == *"ca"* ]];
then

for n in 1 10 #100 1000 4000
do
qsub sgan_ca.qsub $n v
qsub sgan_ca.qsub $n t
done


else

for n in 100 1000 4000
do
qsub sgan_nr.qsub -v "N=$n, model=r"
qsub sgan_nr.qsub -v "N=$n, model=s"
done

fi
