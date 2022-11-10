CUDA_VISIBLE_DEVICES=0 python train_bt.py \
    --task_type em \
    --task doduo \
    --logdir result_doduo/ \
    --ssl_method simclr \
    --clustering \
    --batch_size 64 \
    --lr 5e-5 \
    --lm roberta \
    --n_ssl_epochs 3 \
    --n_epochs 20 \
    --max_len 128 \
    --projector 768 \
    --size 1000 \
    --da cutoff \
    --fp16 \
    --run_id 0

