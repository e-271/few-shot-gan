#!/bin/bash

# 182 is the cute little girl
# 192,100,147,306,225
# Women: 1284(ponytail),1467(firefighter),1702(sunglasses),1690(older)
# Men: 1127(curly),1333(older),1439(kiddo),1554(vet),1995(clean),1943(hat),1636(buisiness)
#seeds=1284,1995
seeds=182
gpu=0
#pkl='/work/newriver/erobb/pickles/car-config-f.pkl'
pkl='/work/newriver/erobb/pickles/ffhq-config-f.pkl'
#pkl='/work/newriver/erobb/pickles/cat-config-f.pkl'

CUDA_VISIBLE_DEVICES=$gpu python run_generator_rand.py generate-images --network=$pkl --seeds=$seeds --truncation-psi=0.6

