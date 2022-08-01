# example script 
CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 path/to/imagenet --model resnet50 --weight_dropout 0.5 --opt_dropout 0.3 --nograd_cutoff 0.01 --temperature 3 --rho 0.05 --epochs 300 --batch-size 256 --lr 0.4 --sched cosine --weight-decay 1e-4 --smoothing 0.0 --model-ema 

--weight_dropout = 1 - beta 
--opt_dropoout = 1 - gamma 

