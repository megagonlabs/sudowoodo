# Sudowoodo

Implementation of the paper "Sudowoodo: Contrastive Self-supervised Learning for End-to-End Data Integration" accepted by IEEE ICDE 2023. A preprint of the paper is available [on arxiv](https://arxiv.org/abs/2207.04122) now.

Sudowoodo is a data integration framework based on contrastive representation learning. Sudowoodo learns similarity-aware data representations from a large corpus of data items (e.g., entity entries, table columns) without using any labels. The learned representations can later be used in components of a DI pipeline either directly or by fine-tuning using only a few labels. We currently support 3 tasks: Entity Matching, Data Cleaning, and Column type detection.

## Requirements

* Python 3.7.10
* PyTorch 1.9.0+cu111
* Transformers 4.9.2
* NVIDIA Apex (fp16 training)

Install required packages
```
pip install -r requirements.txt
git clone https://github.com/NVIDIA/apex.git
pip install -v --disable-pip-version-check --no-cache-dir ./apex
```

## Data

Please download the datasets from [this link](https://drive.google.com/file/d/1V-QN2nazSrhZONVC4JoFfcUqRKmTYL1z/view?usp=sharing) and create a folder named ``data`` to save it so as to directly use the below commands to reproduce the results. 

## Model Training

To train the baseline model:
```
CUDA_VISIBLE_DEVICES=0 python train_bt.py \
    --task_type em \
    --task Abt-Buy \
    --logdir result_em/ \
    --ssl_method combined \
    --bootstrap \
    --clustering \
    --multiplier 10 \
    --batch_size 64 \
    --lr 5e-5 \
    --lm roberta \
    --n_ssl_epochs 3 \
    --n_epochs 50 \
    --max_len 128 \
    --da cutoff \
    --size 500 \
    --fp16 \
    --run_id 4
```

Parameters:
* ``--task_type``: the type of the task (``em``, Entity Matching or ``cleaning``, Data Cleaning)
* ``--task``: the taskname (e.g., DBLP-ACM, Abt-Buy; see ``data/em`` and ``data/cleaning``)
* ``--logdir``: the path for TensorBoard logging (F1, recall, etc.)
* ``--ssl_method``: self-supervised learning method. We support ``simclr``, ``barlow_twins``, and ``combined``
* ``--bootstrap``: if set, then apply pseduo labeling to collect extra labels for fine-tuning. The number of labels added is specified by ``--multiplier`` (how many times more labels). For unsupervised learning, use the ``--zero`` flag.
* ``--clustering``: if this flag is set, then apply the clustering-based negative sampling optimization
* ``--da``: the data augmentation operators. We currently support base operators ``['del', 'drop_col', 'append_col', 'swap', 'ins', 'shuffle']`` and ``cutoff``.
* ``--size``: the dataset size (optional). If not specified, the entire dataset will be used. For semi-supervised or unsupervised EM, set this value to 500 (the unsupervised version will ignore the labels)
* ``--batch_size``, ``--lr``, ``--max_len``, ``--n_epochs``, ``--n_ssl_epochs``: the batch size, learning rate, max sequence length, the total number of epochs, and the number of epochs for pre-training
* ``--fp16``: whether to use half-precision for training
* ``--lm``: the language model to fine-tune. We currently support distilbert and roberta
* ``--run_id``: the integer ID of the run e.g., {0, 1, 2, ...}

To save/load the pre-trained checkpoint, use the flag ``--save_ckpt`` and ``--use_saved_ckpt``. The path of the model checkpoint by default is ``{logdir}/{task}/ssl.pt``.

## Blocking for EM

We support two modes for blocking: evaluation mode and candidate generation mode.

The evaluation mode pre-trains the model and compute the recalls and #candidates:
```
CUDA_VISIBLE_DEVICES=0 python train_bt.py \
    --task_type em \
    --task Abt-Buy \
    --logdir result_blocking/ \
    --ssl_method barlow_twins \
    --batch_size 64 \
    --lr 5e-5 \
    --lm distilbert \
    --n_ssl_epochs 5 \
    --n_epochs 5 \
    --max_len 128 \
    --projector 4096 \
    --da del \
    --blocking \
    --fp16 \
    --save_ckpt \
    --k 20 \
    --run_id 0
```

Note that the ``--blocking`` flag is set and the extra hyper-parameter ``--k 20`` indicates using top-k nearest neighbor search on the right table to generate candidate pairs for k=1 to 20.

To generate the candidate pairs:

```
CUDA_VISIBLE_DEVICES=0 python blocking.py \
    --task Abt-Buy \
    --logdir result_blocking \
    --batch_size 512 \
    --max_len 128 \
    --projector 4096 \
    --lm distilbert \
    --fp16 \
    --k 20 \
    --ckpt_path result_blocking/Abt-Buy/ssl.pt 
```

Parameters:
* ``--ckpt_path``: the path to the checkpoint obtained by setting ``--save_ckpt`` during training
* ``--k``: the top-k nearest neighbor
* ``--threshold`` (optional): the cosine similarity threshold (e.g., 0.6)

The model configurations (e.g., max_len, lm, projector) have to be the same as the training configurations

## Data Cleaning

Our framework also supports the data cleaning task. Please see ``cleaning_example_command.sh`` for reference. More details can be found in Section V of the paper. We provide four datasets in ``data/cleaning/`` for evaluation: beers, hospital, rayyan, and tax.


## Column type detection

See ``column_type_detection/``.

## Citation
If you are using the code in this repo, please cite the following in your work:
```
@inproceedings{icde23sudowoodo,
  author    = {Runhui Wang and
               Yuliang Li and
               Jin Wang},
  title     = {Sudowoodo: Contrastive Self-supervised Learning for End-to-End Data Integration},
  booktitle = {ICDE},
  year      = {2023}
}
```

## Disclosure

Embedded in, or bundled with, this product are open source software (OSS) components, datasets and other third party components identified below. The license terms respectively governing the datasets and third-party components continue to govern those portions, and you agree to those license terms, which, when applicable, specifically limit any distribution. You may receive a copy of, distribute and/or modify any open source code for the OSS component under the terms of their respective licenses. In the event of conflicts between Megagon Labs, Inc. Recruit Co., Ltd., license conditions and the Open Source Software license conditions, the Open Source Software conditions shall prevail with respect to the Open Source Software portions of the software. 
You agree not to, and are not permitted to, distribute actual datasets used with the OSS components listed below. You agree and are limited to distribute only links to datasets from known sources by listing them in the datasets overview table below. You are permitted to distribute derived datasets of data sets from known sources by including links to original dataset source in the datasets overview table below. You agree that any right to modify datasets originating from parties other than Megagon Labs, Inc. are governed by the respective third party’s license conditions. 
All OSS components and datasets are distributed WITHOUT ANY WARRANTY, without even implied warranty such as for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, and without any liability to or claim against any Megagon Labs, Inc. entity other than as explicitly documented in this README document. You agree to cease using any part of the provided materials if you do not agree with the terms or the lack of any warranty herein.
While Megagon Labs, Inc., makes commercially reasonable efforts to ensure that citations in this document are complete and accurate, errors may occur. If you see any error or omission, please help us improve this document by sending information to contact_oss@megagon.ai.

## Contact

If you have any questions regarding the code and the paper, please directly contact Runhui Wang (wangrunhui.pku@gmail.com).
