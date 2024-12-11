# HAKES client package

It include the client to interact with HAKES and also the Sharded-HNSW baseline which features a per node HNSW serving architecture.

## ClientV2

Interact with the Sharded-HNSW deployment.

* Search: it issue requests to all the nodes and receive the results then aggregate locally.
* Write: issue request to the corresponding shard.

## ClientV3

Interact with HAKES

* Search: choose a IndexWorker to get the candidate points; then build batches for RefineWorkers; then aggregate locally.
* Write: write to a pair of IndexWorker and RefineWorker, then propagate the intermediate results to other IndexWorkers.

## Checkpoint

* Issue requests to all servers and let them to checkpoint the index and data locally.

## Install

```sh
pip install .
```

## Uninstall

```sh
pip uninstall hakes_client
```
