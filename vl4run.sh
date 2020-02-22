#!/bin/bash


cfg=$1
gpu1=$2
gpu2=$3

CUDA_VISIBLE_DEVICES=$gpu1,$gpu2 nohup python run_training.py --num-gpus=2 --data-dir=./datasets --config=config-$1 --dataset=kannada --total-kimg=10000 --resume-pkl=./pickles/eng-config-f-10M.pkl > $1.out 2>&1
