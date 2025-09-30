/*
 * Copyright 2024 The HAKES Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/ext/index_io_ext.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

#include "utils/data_loader.h"

namespace {

// benchmark configuration
struct Config {
  uint64_t data_n;
  uint64_t data_num_query;
  uint64_t data_dim;
  std::string data_train_path;
  // index params
  uint64_t opq_out = 32;
  uint64_t m = 32;
  uint64_t nlist = 1024;
  faiss::MetricType metric = faiss::METRIC_INNER_PRODUCT;
  std::string index_save_path = "saved_index";
};

Config parse_config(int argc, char** argv) {
  Config cfg;
  if (argc < 9) {
    std::cout << "Usage: " << argv[0]
              << " DATA_N DATA_DIM DATA_TRAIN_PATH OPQ_OUT M "
                 "NLIST METRIC INDEX_SAVE_PATH"
              << std::endl;
    exit(-1);
  }

  cfg.data_n = std::stoul(argv[1]);
  cfg.data_dim = std::stoul(argv[2]);
  cfg.data_train_path = argv[3];
  cfg.opq_out = std::stoul(argv[4]);
  cfg.m = std::stoul(argv[5]);
  cfg.nlist = std::stoul(argv[6]);
  cfg.metric = (std::stoul(argv[7]) == 0) ? faiss::METRIC_INNER_PRODUCT
                                          : faiss::METRIC_L2;
  cfg.index_save_path = argv[8];

  // print summary
  std::cout << "DATA_N: " << cfg.data_n << std::endl;
  std::cout << "DATA_NUM_QUERY: " << cfg.data_num_query << std::endl;
  std::cout << "DATA_DIM: " << cfg.data_dim << std::endl;
  std::cout << "DATA_TRAIN_PATH: " << cfg.data_train_path << std::endl;
  std::cout << "OPQ_OUT: " << cfg.opq_out << std::endl;
  std::cout << "M: " << cfg.m << std::endl;
  std::cout << "NLIST: " << cfg.nlist << std::endl;
  std::cout << "INDEX_SAVE_PATH: " << cfg.index_save_path << std::endl;

  // find the last slash in the train path and use the rest as the prefix
  auto last_slash = cfg.data_train_path.find_last_of("/");
  if (last_slash == std::string::npos) {
    last_slash = -1;
  }
  auto second_last_slash =
      cfg.data_train_path.substr(0, last_slash).find_last_of("/");
  if (second_last_slash == std::string::npos) {
    second_last_slash = -1;
  }
  std::string dataset_name = cfg.data_train_path.substr(
      second_last_slash + 1, last_slash - second_last_slash - 1);

  cfg.index_save_path = cfg.index_save_path;

  std::cout << "Index path: " << cfg.index_save_path << std::endl;

  return cfg;
}
}  // namespace

int main(int argc, char** argv) {
  auto cfg = parse_config(argc, argv);
  // dataset
  int n = cfg.data_n;
  int d = cfg.data_dim;
  // configuration
  int nlist = cfg.nlist;
  int default_bbs = 32;
  bool use_multithread = false;
  // bool use_multithread = true;

  // load data
  float* data = nullptr;
  data = load_data(cfg.data_train_path.c_str(), d, n);
  faiss::MetricType metric = cfg.metric;

  // create ivf pq index
  auto build_start = std::chrono::high_resolution_clock::now();

  faiss::IndexFlat quantizer{cfg.opq_out, metric};
  faiss::IndexIVFPQ base_index{&quantizer, cfg.opq_out, nlist,
                               cfg.m,      4,           metric};
  base_index.by_residual = false;  // reclaim the quantizer

  faiss::IndexPreTransform index(&base_index);
  faiss::OPQMatrix opq_vt{d, cfg.m, cfg.opq_out};
  index.prepend_transform(&opq_vt);

  // train
  index.train(n, data);
  // add
  auto build_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> build_diff = build_end - build_start;
  std::cout << "train time (s): " << build_diff.count() << std::endl;
  std::unique_ptr<faiss::FileIOWriter> ff =
      std::make_unique<faiss::FileIOWriter>(cfg.index_save_path.c_str());
  faiss::save_init_params(ff.get(), &index.chain, &base_index.pq, &quantizer);

  // clean up
  delete[] data;
  return 0;
}
