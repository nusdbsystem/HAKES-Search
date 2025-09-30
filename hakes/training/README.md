# HAKES index train

HAKES lightweght self-supervised training to generate the learned compression parameters for query via jointly optimization of dimension reduction and quantization.

## Prepare training datasets

The training dataset can be constructed from a sample set of query vectors and their groundtruth neighbors. But it is not necessary, sampling vectors and their ANNs from the indexed dataset with the base index parameters is shown to be efficient alternative in our paper. Use the `hakes/index/gen_trainset.cpp` to generate such training dataset. (For OOD scenarios, using sample query set is highly beneficial)

```sh
./gen_trainset ${N} ${d} 100 <data-path> 0 <base-index-dir>
# base-index-dir is the directory that contains findex.bin
```

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

There are scripts for exploring different parameters under `scripts`. An example command is:

```sh
python main.py --N 1000000 --d 768 --train_n 100000 --train_neighbor_count 100 --use_train_nn_count 50 --NQ 1000 --data_path ~/data/sphere-768/ --index_path ../index/base_index/ --epoch 1 --batch_size 512 --vt_lr 1e-4 --pq_lr 1e-4 --lamb -1 --device cuda:1 --save_path <output_path> > log.txt
```

test the learned search index parameters with `hakes/index/hakes_index.cpp`. Note that we can either place the `uindex.bin` generated under the `index_path` that contains the `findex.path` or passing the `uindex.bin` path as an additional argument when executing `hakes_index`.

```sh
./hakes_index <N> <NQ> <d> <k> <groundtruth_len> <data_path> <query_path> <groundtruth_path> 1 1 1 0 1 100 50 0 <index_dir>
./hakes_index <N> <NQ> <d> <k> <groundtruth_len> <data_path> <query_path> <groundtruth_path> 1 1 1 0 1 100 50 0 <index_dir> <uindex_bin_path>
```

## uninstall

```sh
pip uninstall hakes
```
