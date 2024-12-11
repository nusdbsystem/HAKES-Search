### Project Settings ###
PROJECT_ROOT_DIR ?= $(shell readlink -f .)
HAKES_ROOT_DIR ?= $(PROJECT_ROOT_DIR)/..

FAISS_ROOT = ../install
FAISS_INCLUDE_PATH = $(FAISS_ROOT)/include
FAISS_LIBRARY_PATH = $(FAISS_ROOT)/lib
FAISS_FLAGS = -I$(FAISS_INCLUDE_PATH) -L$(FAISS_LIBRARY_PATH) -lfaiss_avx2
MKL_LIBRARY_PATH = $(HOME)/intel/oneapi/mkl/2024.0/lib/
MKL_FLAGS = -L${MKL_LIBRARY_PATH} -lmkl_rt

DEBUG_FLAGS = -g -fsanitize=address 

.PHONY: all

all: hakes_index gen_base_params

gen_base_params: gen_base_params.cpp
	g++ -Ofast -lrt -std=c++17 -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 $^ -o $@ -I. -I${HAKES_ROOT_DIR} ${FAISS_FLAGS} ${MKL_FLAGS} -pthread -lm -fopenmp

hakes_index: hakes_index.cpp
	g++-11 -Ofast -lrt -std=c++17 -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 $^ -o $@ -I. -I${HAKES_ROOT_DIR} ${FAISS_FLAGS} ${MKL_FLAGS} -pthread -lm -fopenmp

clean:
	rm -rf *.o hakes_index gen_base_params
