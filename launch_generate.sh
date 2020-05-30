#!/bin/bash

# seeds="100-200"
seeds="116"
#dir="/mnt/hdd/erobb/plots/table1/kannada4K/00008-stylegan2-kannada4K-2gpu-config-f-25img-rho0.0E00-lr2.0E-04/"
#dir="/mnt/hdd/erobb/plots/table1/kannada4K/00005-stylegan2-kannada4K-2gpu-config-f-10img-rho0.0E00-lr2.0E-04"
#dir="/mnt/hdd/erobb/plots/table1/kannada4K/00008-stylegan2-kannada4K-2gpu-config-f-25img-rho0.0E00-lr2.0E-04"
#dir=$1
# pkl=$1
# pkl="/work/newriver/jiaruixu/results/arch_abl/obama25_31shot/pca/00000-stylegan2-obama25-obama25-2gpu-config-pc-all-spc-31img-aug-0sv/network-snapshot-000020.pkl"
pkl="/work/newriver/jiaruixu/results/personalize/rem25_25shot/pca/00004-stylegan2-rem25-rem25-1gpu-config-pc-all-spc-25img-aug-0sv/network-snapshot-000020.pkl"
# pkl="/work/newriver/jiaruixu/results/personalize/4978_31shot/pca/00002-stylegan2-4978-4978-1gpu-config-pc-all-spc--31img-aug-0sv/network-snapshot-000020.pkl"
gpu=$1
l=3

CUDA_VISIBLE_DEVICES=$gpu python run_generator.py generate-images --network=$pkl --seeds=$seeds --layer-toggle=$l
# python run_generator.py generate-images --network=$pkl --seeds=$seeds --layer-toggle=$l --layer-dset=obama25 --layer-ddir=/work/newriver/erobb/datasets/mini/faces

# trap "exit" INT
#/mnt/hdd/erobb/results/arch_abl/4978_31shot/t/00001-stylegan2-4978-4978-1gpu-config-f-31img-aug/network-snapshot-000000.pkl /mnt/hdd/erobb/results/arch_abl/4978_31shot/t/00001-stylegan2-4978-4978-1gpu-config-f-31img-aug/network-snapshot-000012.pkl /mnt/hdd/erobb/results/arch_abl/4978_31shot/pca/00002-stylegan2-4978-4978-1gpu-config-pc-all-spc--31img-aug-0sv/network-snapshot-000016.pkl
# for pkl in /mnt/hdd/erobb/results/arch_abl/3719_31shot/pca/00004-stylegan2-3719-3719-1gpu-config-pc-all-spc-31img-aug-0sv/network-snapshot-000015.pkl /mnt/hdd/erobb/results/arch_abl/3719_30shot/fd/00000-stylegan2-3719-3719-1gpu-config-f-30img-aug/network-snapshot-000015.pkl /mnt/hdd/erobb/results/arch_abl/3719_30shot/fd/00000-stylegan2-3719-3719-1gpu-config-f-30img-aug/network-snapshot-000000.pkl
# do
# python run_generator.py generate-images --network=$pkl --seeds=$seeds --layer-toggle=$l --layer-dset=tower25 --layer-ddir=/mnt/slow_ssd/erobb/datasets
# done
