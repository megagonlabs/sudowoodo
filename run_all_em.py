datasets = """Amazon-Google
DBLP-ACM
DBLP-GoogleScholar
Walmart-Amazon
Abt-Buy""".split('\n')

ssl_eps = [3, 3, 3, 3, 3]
multipliers = [8, 8, 8, 8, 8]

result_path = 'results_em_combined/'
sizes = [500]
lm = 'roberta'
da = 'cutoff'
ssl_method = 'combined'
clustering = True
batch_size = 64
n_epochs_list = [70]

# if clustering:
#     multipliers = [8]
# else:
#     multipliers = [5, 5, 5, 5, 4]

import os
import time

# os.system("export MLFLOW_EXPERIMENT_NAME=sudowoodo")

# parameters = [("combined", "cutoff", True, "bootstrap"),
#             ("mtl", "cutoff", True, "bootstrap"),
#             ("simclr", "cutoff", True, "bootstrap"),
#             ("barlow_twins", "cutoff", True, "bootstrap"),
#             ("combined", "del", True, "bootstrap"),
#             ("combined", "cutoff", False, "bootstrap"),
#             ("combined", "cutoff", True, "None")]

parameters = [("simclr", "del", False, "None"),
              ("simclr", "del", False, "bootstrap"),
              ("simclr", "del", True, "bootstrap")]

# DM + RoBERTa
for param in parameters:
    ssl_method, da, clustering, setting = param

    for dataset, ssl_ep, mul in zip(datasets, ssl_eps, multipliers):
        for n_epochs, size in zip(n_epochs_list, sizes):
            if ssl_method == 'mtl' or setting == 'None':
                n_epochs = 30

            for run_id in range(5):
                # cmd = """CUDA_VISIBLE_DEVICES=1 python train_bt.py \
                cmd = """python train_bt.py \
        --mlflow_tag ablation \
        --task_type em \
        --task %s \
        --ssl_method %s \
        --multiplier %d \
        --logdir %s \
        --batch_size %d \
        --lr 5e-5  \
        --lm %s \
        --n_ssl_epochs %d \
        --n_epochs %d \
        --max_len 128 \
        --projector 768 \
        --da %s \
        --fp16 \
        --size %d \
        --run_id %d""" % (dataset, ssl_method, mul, result_path, batch_size, lm, ssl_ep,
                      n_epochs, da, size, run_id)
                if setting in ['bootstrap', 'zero']:
                    cmd += ' --%s' % setting
                if clustering:
                    cmd += ' --clustering'
                # if run_id == 0:
                #     cmd += ' --save_ckpt'
                # else:
                #     cmd += ' --use_saved_ckpt'
                print(cmd)
                os.system('sbatch -c 1 -G 1 -J my-exp --tasks-per-node=1 --wrap="%s"' % cmd)
                # os.system(cmd)

# python create_tasks.py | xargs -I {} sbatch -c 1 -G 1 -J my-exp --tasks-per-node=1 --wrap="{}"

