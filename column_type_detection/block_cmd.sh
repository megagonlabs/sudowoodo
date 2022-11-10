CUDA_VISIBLE_DEVICES=0 python blocking.py \
    --task doduo \
    --logdir result_doduo/ \
    --batch_size 512 \
    --ckpt_path result_doduo/doduo/ssl.pt \
    --lm roberta \
    --max_len 128 \
    --projector 768 \
    --k 20 \
    --fp16 \
    --run_id 0

