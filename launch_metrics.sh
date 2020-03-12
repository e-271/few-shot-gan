#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: ./metrics.sh file.out"
fi

output=$1
#dir="/mnt/hdd/erobb/plots/table1/kannada4K/00005-stylegan2-kannada4K-2gpu-config-f-10img-rho0.0E00-lr2.0E-04"
#dir="/mnt/hdd/erobb/plots/table1/kannada4K/00008-stylegan2-kannada4K-2gpu-config-f-25img-rho0.0E00-lr2.0E-04"
#dir="/mnt/hdd/erobb/plots/table1/kannada4K/00004-stylegan2-kannada4K-2gpu-config-c-b-10img-rho0.0E00-lr2.0E-04"
#dirs="/mnt/hdd/erobb/plots/table3/dog25/*/"

ddir="/work/newriver/erobb/datasets/evals"


#dirs="/work/newriver/erobb/results/table3/tower25/*/"
#dset="towers"

#dirs=$1
#dirs="/work/newriver/erobb/results/table3/anime25/00022-stylegan2-anime25-2gpu-config-a-gb-25img-rho0.0E00-lr2.0E-04"
#dirs="/work/newriver/erobb/results/table3/dog25/00002-stylegan2-dog25-2gpu-config-a-gb-25img-rho0.0E00-lr2.0E-04/"
#dirs="/work/newriver/erobb/results/table3/tower25/00011-stylegan2-tower25-2gpu-config-a-gb-25img-rho0.0E00-lr2.0E-04/"
#dirs="/work/newriver/erobb/results/table3/anime25/00022-stylegan2-anime25-2gpu-config-a-gb-25img-rho0.0E00-lr2.0E-04"
#dirs="/work/newriver/erobb/results/table3/tower25/00001-stylegan2-tower25-2gpu-config-f-25img-rho0.0E00-lr2.0E-04"
dirs=$1 
#dirs="/work/newriver/erobb/results/table3/tower25/000[0-1][0-1]*"
#"/work/newriver/erobb/results/table3/dog25/00001-stylegan2-dog25-2gpu-config-f-25img-rho0.0E00-lr2.0E-04"
#dset=$2
#dset="danbooru1024" 
#dset="dogs"
#dset="towers"
dset=$2

for dir in `ls -d $dirs`
do

for pkl in $dir/network*
do
echo $pkl
python run_metrics.py --network $pkl --metrics "pps,fid1k" --dataset $dset --data-dir $ddir | tee -a $dir/pps_fid1k.out
done

done
