
export CUDA_VISIBLE_DEVICES=0
python src/main_incremental.py --approach vlwf --nepochs 200 --batch-size 128 --num-workers 4 --datasets tiny --num-tasks 20 --nc-first-task 10 --lr 0.05 --weight-decay 5e-4 --clipping 1  --network bbbresnet18  --momentum 0.9 --exp-name exp1 --seed 0

