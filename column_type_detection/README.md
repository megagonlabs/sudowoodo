## Sudowoodo for Column Type Detection

### Dataset

We conducted the column type detection experiment using the VizNet dataset obtained from [Sato](https://github.com/megagonlabs/sato). The datasets are preprocessed and stored in the files ``data/sato_cv_{0,1,2,3,4}.csv`` where each file is 1/5 of the original dataset and each row in the csv file represents a serialized table column. The original table files are in ``data/sato_tables.tar.gz``.

We also provide the intermediate blocking and matching results. After unzipping ``intermediate_results.tar.gz``, we have 4 files:
* ``columns_labeled.txt``: all the serialized columns (one per line) with class names and ID's
* ``columns.txt``: the serialized columns (one per line) without labels
* ``blocking_result.pkl``: a python list of all candidate pairs. Each pair is of the format ``(i, j, score)`` which indicates that the i-th column and the j-th column is a candidate match.
* ``test_results.pkl``: a python integer list indicating whether each pair is match (1) or not (0).

### Data preparation

The python code ``create_blocking_input.py`` is for generating the files ``columns.txt`` and ``columns_labeled.txt``. Simply run:
```
python create_blocking_input.py
cp data/columns.txt ../data/em/doduo/tableA.txt
cp data/columns.txt ../data/em/doduo/tableB.txt
cp data/columns.txt ../data/em/doduo/train_no_label.txt
```

### Blocking 

First, we need to pre-train the representation model:

```
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
    --n_epochs 4 \
    --max_len 128 \
    --projector 768 \
    --size 1000 \
    --da cutoff \
    --save_ckpt \
    --fp16 \
    --run_id 0
```

The command will create a checkpoint in ``result_doduo/doduo/ssl.pt``. We can ignore the matching output for now.

We are now ready to generate the candidate pairs. Use the command in ``block_cmd.sh`` under the root directory:

```
./column_type_detection/block_cmd.sh
```

This command will create the file ``blocking_result.pkl``. This command takes about 15 minutes with an A100 GPU.

### Matching

To run matching, we first need to create a labeled dataset. We do so by using Sato's groundtruth labels. Run the command:

```
python create_matching_datasets.py
cp train.txt valid.txt test.txt all_pairs.txt ../data/em/doduo/
```

Here ``train/valid/test.txt`` are the created training, validation, and test sets. The file ``all_pairs.txt`` contains all candidate pairs for running the matching model.

We can now run the matching model training and prediction:

```
./column_type_detection/match_cmd.sh
```

Tis command will generate the file ``test_results.pkl``.

### Clustering

We use the jupyter notebook ``visualize.ipynb`` for computing the connected components and inspecting clusters.
