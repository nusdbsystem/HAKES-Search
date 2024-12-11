PROJECT_ROOT_DIR ?= $(shell readlink -f ../..)
FAISS_INSTALL_DIR ?= $(PROJECT_ROOT_DIR)/install
INSTALL_PREFIX ?= $(shell readlink -f ../../install)
MKL_LIBRARY_PATH ?= ${HOME}/intel/oneapi/mkl/2024.0/lib/
HNSW_INCLUDE_PATH ?= $(PROJECT_ROOT_DIR)/../../annlib/hnswlib

DEPS_INSTALL_DIR = $(PROJECT_ROOT_DIR)/deps/install
LIBUV_DIR = $(DEPS_INSTALL_DIR)/libuv
LLHTTP_DIR = $(DEPS_INSTALL_DIR)/llhttp

CONCURRENT_RT_CXX_FLAGS := -I$(LIBUV_DIR)/include
CONCURRENT_RT_LINK_FLAGS := -L$(LIBUV_DIR)/lib -l:libuv_a.a
CONCURRENT_RT_CXX_FLAGS += -I$(LLHTTP_DIR)/include
CONCURRENT_RT_LINK_FLAGS += -L$(LLHTTP_DIR)/lib -l:libllhttp.a

HAKESINDEX_CXX_FLAGS += -I$(FAISS_INSTALL_DIR)/include
HAKESINDEX_LINK_FLAGS += -L$(FAISS_INSTALL_DIR)/lib -lfaiss_avx2 -L$(MKL_LIBRARY_PATH) -lmkl_rt

HNSW_CXX_FLAGS += -I$(HNSW_INCLUDE_PATH)

CONCURRENT_RT_CXX_FLAGS += -I${PROJECT_ROOT_DIR}

deps:
	@echo "run make deps under repo root to build deps"

# build with the Makefile under concurrent_runtime

COMMON_CXX_FLAGS := $(CONCURRENT_RT_CXX_FLAGS) -I$(PROJECT_ROOT_DIR)/serving

.PHONY: all clean install

all: hakes-shard-server hnsw-shard-server

# setup compile flags
COMMON_CXX_FLAGS += -Wno-attributes
# setup faiss compile flags
COMMON_CXX_FLAGS += -Ofast -std=c++17 -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -g

COMMON_LINK_FLAGS += $(CONCURRENT_RT_LINK_FLAGS)
# setup faiss link flags
COMMON_LINK_FLAGS += -lrt -pthread -lm -fopenmp -ldl

Common_CXX_Flags += -Wall -Wextra -Winit-self -Wpointer-arith -Wreturn-type \
                    -Waddress -Wsequence-point -Wformat-security \
                    -Wmissing-include-dirs -Wfloat-equal -Wundef -Wshadow \
                    -Wcast-align -Wcast-qual -Wconversion -Wredundant-decls

Source_objects = $(Sources:.cpp=.o)

%.o: %.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

json.o: $(PROJECT_ROOT_DIR)/utils/json.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

hexutil.o: $(PROJECT_ROOT_DIR)/utils/hexutil.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

checkpoint.o: ${PROJECT_ROOT_DIR}/serving/common/checkpoint.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

message.o: $(PROJECT_ROOT_DIR)/message/message.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

message_ext.o: $(PROJECT_ROOT_DIR)/message/message_ext.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

server.o: $(PROJECT_ROOT_DIR)/serving/server/server.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

service.o: $(PROJECT_ROOT_DIR)/serving/server/service.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

hnswshardengine.o: $(PROJECT_ROOT_DIR)/serving/shard-worker/hnswshardengine.cpp
	$(CXX) $(COMMON_CXX_FLAGS) $(HNSW_CXX_FLAGS) -c $< -o $@

hakesshardengine.o: $(PROJECT_ROOT_DIR)/serving/shard-worker/hakesshardengine.cpp
	$(CXX) $(COMMON_CXX_FLAGS) $(HAKESINDEX_CXX_FLAGS) -c $< -o $@

worker.o: $(PROJECT_ROOT_DIR)/serving/shard-worker/worker.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

Source_objects += json.o hexutil.o checkpoint.o message.o message_ext.o server.o service.o worker.o

hnsw_shard_main.o: $(PROJECT_ROOT_DIR)/serving/shard-worker/hnsw_shard_main.cpp
	$(CXX) $(COMMON_CXX_FLAGS) $(HNSW_CXX_FLAGS) -c $< -o $@

hakes_shard_main.o: $(PROJECT_ROOT_DIR)/serving/shard-worker/hakes_shard_main.cpp
	$(CXX) $(COMMON_CXX_FLAGS) $(HAKESINDEX_CXX_FLAGS) -c $< -o $@

hnsw-shard-server: hnsw_shard_main.o hnswshardengine.o $(Source_objects)
	$(CXX) $^ -o $@ $(COMMON_LINK_FLAGS)

hakes-shard-server: hakes_shard_main.o hakesshardengine.o $(Source_objects)
	$(CXX) $^ -o $@ $(HAKESINDEX_LINK_FLAGS) $(COMMON_LINK_FLAGS)

install:
	install -d $(INSTALL_PREFIX)/bin
	install -C -m 755 hakes-shard-server $(INSTALL_PREFIX)/bin/hakes-shard-server
	install -C -m 755 hakes-shard-server $(INSTALL_PREFIX)/bin/hnsw-shard-server

clean:
	rm -f *.o
	rm -f hakes-shard-server hnsw-shard-server

mrproper: clean
	rm -rf install
