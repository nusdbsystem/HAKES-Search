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

CONCURRENT_RT_CXX_FLAGS += -I$(FAISS_INSTALL_DIR)/include
CONCURRENT_RT_LINK_FLAGS += -L$(FAISS_INSTALL_DIR)/lib -lfaiss_avx2 -L$(MKL_LIBRARY_PATH) -lmkl_rt

CONCURRENT_RT_CXX_FLAGS += -I${PROJECT_ROOT_DIR}

deps:
	@echo "run make deps under repo root to build deps"

# build with the Makefile under concurrent_runtime

COMMON_CXX_FLAGS := $(CONCURRENT_RT_CXX_FLAGS) -I$(PROJECT_ROOT_DIR)/serving

.PHONY: all clean install

all: hakes-server hakes-be-server

# setup compile flags
# SANITIZER_FLAGS := -fsanitize=address,undefined
COMMON_CXX_FLAGS += -Wno-attributes
# setup faiss compile flags
COMMON_CXX_FLAGS += -O2 -std=c++17 -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -g $(SANITIZER_FLAGS)
# COMMON_CXX_FLAGS += -Ofast -std=c++17 -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -g

COMMON_LINK_FLAGS += $(CONCURRENT_RT_LINK_FLAGS)
# setup faiss link flags
COMMON_LINK_FLAGS += -lrt -pthread -lm -fopenmp -ldl $(SANITIZER_FLAGS)

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

hakesexecutor.o: $(PROJECT_ROOT_DIR)/serving/fnr-worker/hakesexecutor.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

hakesengine.o: $(PROJECT_ROOT_DIR)/serving/fnr-worker/hakesengine.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

task.o: $(PROJECT_ROOT_DIR)/serving/task-queue/task.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

hakestask.o: $(PROJECT_ROOT_DIR)/serving/fnr-worker/hakestask.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

hakesbatchengine.o: $(PROJECT_ROOT_DIR)/serving/fnr-worker/hakesbatchengine.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

worker.o: $(PROJECT_ROOT_DIR)/serving/fnr-worker/worker.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

Source_objects += json.o hexutil.o checkpoint.o message.o message_ext.o server.o service.o worker.o

hakes_main.o: $(PROJECT_ROOT_DIR)/serving/fnr-worker/hakes_main.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

hakes_be_main.o: $(PROJECT_ROOT_DIR)/serving/fnr-worker/hakes_be_main.cpp
	$(CXX) $(COMMON_CXX_FLAGS) -c $< -o $@

hakes-server: hakes_main.o hakesexecutor.o hakesengine.o $(Source_objects)
	$(CXX) $^ -o $@ $(COMMON_LINK_FLAGS)

hakes-be-server: hakes_be_main.o hakesexecutor.o task.o hakestask.o hakesbatchengine.o $(Source_objects)
	$(CXX) $^ -o $@ $(COMMON_LINK_FLAGS)

install:
	install -d $(INSTALL_PREFIX)/bin
	install -C -m 755 hakes-server $(INSTALL_PREFIX)/bin/server

clean:
	rm -f *.o test/*.o
	rm -f hakes-server hakes-be-server
	rm -f test/hakesengine_test test/hakesbatchengine_test

mrproper: clean
	rm -rf install
