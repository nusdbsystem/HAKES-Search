# HAKES-Search

HAKES: Scalable Vector Database for Embedding Search Service

This code repository is a cleaned version prepared for VLDB2025 submission.

## Extended version

The folder [./extended-version](./extended-version/) contains the extended version of the submission with additional discussion and experimental results.

## Coming soon

We will release the codes and instructions to prepare the experiment data and trained index parameters soon.

## Code structure

HAKES repo include the following parts

* `faiss-ex`: An extended faiss library for building index used in HAKES to serve concurrent read-write requests. We try to isolate most of our extensions under `faiss-ex/faiss/ext`, but there are some direct modification to faiss source files for implementation convenience. While we try to keep the rest of faiss library unmodified unless it is necessary to expose certain functions or structs, and register hooks to use our implementations.
* `hakes`: the index enhancement module, serving system, and clients

  * `training`: the light-weight trainig process to enhance the recall used in HAKES.
  * `serving`: the server implementation for serving vector requests with the index built with our extended `faiss-ex` and enhanced with `training`.
  * `client`: the python client to interact with HAKES.

Both `training` and `client` can be easily installed to the local environment with pip. Please refer to the README under the respective directories.

## Building faiss-ex

It follows the same build procedures for faiss. We added our extension files to the faiss cmake file, such that the built library includes our extension codes.

Faiss depends on Intel oneMKL. Please download and install Intel oneMKL from <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&distributions=online> first.

```sh
cd faiss-ex
mkdir build
cmake -B build -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx2 -DBLA_VENDOR=Intel10_64_dyn -DCMAKE_INSTALL_PREFIX=../hakes/install -DMKL_LIBRARIES=$HOME/intel/oneapi/mkl/2024.0/lib/libmkl_rt.so .
make -C build install -j4
```

It shall install the built binary to `hakes/install` and HAKES will link with the `install/lib/libfaixx_avx2.so`.

## Building HAKES serving

Dependecies of HAKES serving includes `libuv` and `llhttp`. We add them as submodule such that they are automatically installed while we build the HAKES serving module and the servers.

```sh
cd hakes
make preparation # it will initialize the dependencies of libuv and llhttp
make server_deps # build libuv and llhttp
# build the server executables
cd serving/fnr-worker
make all
```

## Use HAKES

### prepare the base index

We provide the programs to generate the base parameters at `hakes/index/gen_base_params.cpp`.

```sh
cd hakes/index
make all
```

```sh
mkdir index_test
./gen_base_params <N> <d> <data_path> <opq_out> <pq_m> <ivf_nlist> <metric> <index_save_path> 
# example: ./gen_base_params 1000000 768 sphere-768/train.bin 192 96 1024 0 index_test/findex.bin
# metric: 0 for ip and 1 for l2
```

The generated base parameters is saved as binary file.

Please use `hakes/index/hakes_index.cpp` to instantiate an index on a dataset, evaluate and generate training sample from the index. Check out `hakes_index.cpp` for different way of testing and modify as needed for different range of `nprobe` and `k_factor`.

```sh
./hakes_index <N> <NQ> <d> <k> <groundtruth_len> <data_path> <query_path> <groundtruth_path> 1 1 1 0 1 100 50 0 <index_dir>
```

We provide scripts in `hakes/index/scripts` to demostrate its usage.

### enhance the index

Please refer to the README in `hakes/training` for more details.

### launch HAKES

Please add the extended faiss dynamic library to your loader path.

```sh
UV_THREADPOOL_SIZE=20 ./sample-server-v3 2351 <hakes_serving_dir>
```

The UV_THREADPOOL_SIZE controls the number of threads used for serving requests.

The hakes serving directory shall have the following structure. Where the built and enhanced index folder is added as the initial checkpoint.

```text
.
└── checkpoint_0
    └── base_index
        ├── findex.bin
        ├── rindex.bin
        └── uindex.bin
```

Checkpoint requests trigger HAKES to create additional checkpoints under this directory. And when the server is restarted, it will load the latest checkpoint.

## Reference

Please cite our publication when you use HAKES in your research or development.

* Guoyu Hu, Shaofeng Cai, Tien Tuan Anh Dinh, Zhongle Xie, Cong Yue, Gang Chen, and Beng Chin Ooi. HAKES: Scalable Vector Database for Embedding Search Service. PVLDB, 18(9): 3049 - 3062, 2025. doi:10.14778/3746405.3746427
