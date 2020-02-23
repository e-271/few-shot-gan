
c=$1
n=$2
a=0
lr=0.0002
rr='pickles/
rri=0
ddir='/mnt/slow_ssd/erobb/stylegan2-adaptive/evals'

python run_training.py --num-gpus=2 --data-dir=./datasets --config=config-$c --dataset=kannada4K --total-kimg=1000 --max-images=$n --resume-pkl=$rr --resume-kimg=$rri --rho=$a --lrate-base=$lr --result-dir=

