#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: ./metrics.sh file.out"
fi

output=$1
#dir="/mnt/hdd/erobb/plots/table1/kannada4K/00005-stylegan2-kannada4K-2gpu-config-f-10img-rho0.0E00-lr2.0E-04"
#dir="/mnt/hdd/erobb/plots/table1/kannada4K/00008-stylegan2-kannada4K-2gpu-config-f-25img-rho0.0E00-lr2.0E-04"
#dir="/mnt/hdd/erobb/plots/table1/kannada4K/00004-stylegan2-kannada4K-2gpu-config-c-b-10img-rho0.0E00-lr2.0E-04"
dirs="/mnt/hdd/erobb/plots/table3/dog25/*/"

for dir in `ls -d $dirs`
do

dset="kannada4K"
ddir="/mnt/slow_ssd/erobb/datasets"
rm $dir/pps_lpips_fid5k.out

for pkl in $dir/network*
do
echo $pkl
python run_metrics.py --network $pkl --metrics "pps,lpips,fid5k" --dataset $dset --data-dir $ddir | tee -a $dir/pps_lpips_fid5k.out
done

done
