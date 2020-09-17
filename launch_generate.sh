#!/bin/bash

# 182 is the cute little girl
# 192,100,147,306,225
# Women: 1284(ponytail),1467(firefighter),1702(sunglasses),1690(older)
# Men: 1127(curly),1333(older),1439(kiddo),1554(vet),1995(clean),1943(hat),1636(buisiness)
#seeds=1284,1995
seeds=100-110
#pkl="/work/newriver/erobb/results/personalize/rem25_25shot/pca/00004-stylegan2-rem25-rem25-1gpu-config-pc-all-spc-25img-aug-0sv/network-snapshot-000011.pkl"
#pkl="/work/newriver/erobb/results/paper/rem25_25shot/pca/00004-stylegan2-rem25-rem25-1gpu-config-pc-all-spc-25img-aug-0sv/network-snapshot-000000.pkl"
#pkl="/work/newriver/erobb/results/1shot/obama_25shot/pca/00000-stylegan2---1gpu-config-pc-all-25img-0sv/network-snapshot-000000.pkl"
pkl='/work/newriver/erobb/results/1shot/obama_25shot/pca/00059-stylegan2-obama-obama-1gpu-config-pc-all-25img-0sv/network-snapshot-000000.pkl'
gpu=0

#CUDA_VISIBLE_DEVICES=$gpu python run_generator.py generate-images --network=$pkl --seeds=$seeds --truncation-psi=0.8
CUDA_VISIBLE_DEVICES=$gpu python run_generator.py generate-images --network=$pkl --seeds=$seeds --truncation-psi=0.6
