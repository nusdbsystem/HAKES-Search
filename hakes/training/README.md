# HAKES index train

HAKES lightweght self-supervised training to generate the learned compression parameters for query via jointly optimization of dimension reduction and quantization.

## Install Hakes-train package

```sh
pip install .
# remove the local build file
bash clean.sh
```

## Training

Use `main.py` for training the index.

Explanation of the parameters:

```text
  --N N                 Number of total data
  --d D                 Dimension of vectors
  --train_n TRAIN_N     Number of training data
  --train_neighbor_count TRAIN_NEIGHBOR_COUNT
                        Number of neighbors per train vector in the file
  --use_train_nn_count USE_TRAIN_NN_COUNT
                        Number of top neighbors to use per query out of train_neighbor_count in training
  --NQ NQ               Number of validation set data count
  --data_path DATA_PATH
                        Data path that should contains the overall data and training data (sampled data and their approximate neighbors)
  --index_path INDEX_PATH
                        Path of base index, in which there should be a base_index directory that can be loaded and served by HAKES
  --epoch EPOCH         Number of epoch to train
  --batch_size BATCH_SIZE
                        Batch size for training iteration
  --vt_lr VT_LR         Learning rate for IVF
  --pq_lr PQ_LR         Learning rate for product quantization
  --lamb LAMB           Control the weight of vt loss, -1 to rescale against pq loss
  --device DEVICE       The GPU device to use if any. (use cpu by default) example: cuda:0
  --save_path_prefix SAVE_PATH_PREFIX
                        Path to save the index trained per epoch
```

## uninstall

```sh
pip uninstall hakes
```
