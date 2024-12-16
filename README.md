# HAKES-Search

HAKES: Scalable Vector Database for Embedding Search Service

This code repository is a cleaned version prepared for VLDB2025 submission.

## Extended version

The folder [./extended-version](./extended-version/) contains the extended version of the submission with additional discussion and experimental results.

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
make server
# build the server executables
cd serving/fnr-worker
make 
```

## Use HAKES

### prepare the base index

We provide the programs to generate the base parameters at `hakes/index/gen_base_params.cpp`.

```sh
cd hakes/index
make all
```

```sh
./gen_base_params <N> <d> <data-path> <opq_out> <pq_m> <ivf_nlist> <index_save_path> 
```

The generated base parameters is saved as a directory with the following structure:

```text
.
└── base_index
    ├── ivf.bin
    ├── pq.bin
    └── pre-transform.bin
```

Please use `hakes/index/hakes_index.cpp` to instantiate an index on a dataset, evaluate and generate training sample from the index.

We provide scripts in `hakes/index/scripts` to demostrate its usage.

### enhance the index

Please refer to the README in `hakes/training` for more details.

### launch HAKES

Please add the extended faiss dynamic library to your loader path.

```sh
UV_THREADPOOL_SIZE=20 ./sample-server-v3 2351 <hakes-serving-dir>
```

The UV_THREADPOOL_SIZE controls the number of threads used for serving requests.

The hakes serving directory shall have the following structure. Where the built and enhanced index folder is added as the initial checkpoint.

```text
.
└── checkpoint_0
    └── base_index
        ├── ivf.bin
        ├── pq.bin
        └── pre-transform.bin
```

Checkpoint requests trigger HAKES to create additional checkpoints under this directory. And when the server is restarted, it will load the latest checkpoint.
