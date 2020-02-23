
c=$1
n=$2
a=0
lr=0.0002
rr='pickles/eng-config-f-10M.pkl'
rri=0
ddir='./results' # changeme

python run_training.py --num-gpus=2 --data-dir=./datasets --config=config-$c --dataset=kannada4K --total-kimg=1000 --max-images=$n --resume-pkl=$rr --resume-kimg=$rri --rho=$a --lrate-base=$lr --result-dir=$ddir

