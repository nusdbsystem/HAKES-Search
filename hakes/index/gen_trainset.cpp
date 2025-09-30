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
#include <faiss/VectorTransform.h>
#include <faiss/ext/HakesIndex.h>
#include <faiss/ext/IndexFlatL.h>
#include <faiss/ext/IndexIVFPQFastScanL.h>
#include <faiss/ext/IndexRefineL.h>
#include <faiss/ext/index_io_ext.h>
#include <faiss/ext/utils.h>
#include <omp.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

#include "utils/data_loader.h"

namespace {

std::atomic<int> thread_launched(0);

// benchmark configuration
struct Config {
  size_t data_n;
  size_t data_dim;
  size_t search_k = 20;  // top_k
  std::string data_train_path;
  // index params
  size_t num_client = 1;
  float write_ratio = 0;
  int batch_size = 1;
  int metric_type = 0;

  std::string index_path = "./index";
  std::string update_index = "";
  size_t report_interval = 10000;
};

Config parse_config(int argc, char** argv) {
  Config cfg;
  if (argc < 7) {
    std::cout << "Usage: " << argv[0]
              << "DATA_N DATA_DIM SEARCH_K DATA_TRAIN_PATH METRIC_TYPE INDEX_PATH"
              << std::endl;
    exit(-1);
  }

  cfg.data_n = std::stoul(argv[1]);
  cfg.data_dim = std::stoul(argv[2]);
  cfg.search_k = std::stoul(argv[3]);
  cfg.data_train_path = argv[4];
  cfg.metric_type = std::stoul(argv[5]);
  cfg.index_path = argv[6];

  // print summary
  std::cout << "DATA_N: " << cfg.data_n << std::endl;
  std::cout << "DATA_DIM: " << cfg.data_dim << std::endl;
  std::cout << "SEARCH_K: " << cfg.search_k << std::endl;
  std::cout << "DATA_TRAIN_PATH: " << cfg.data_train_path << std::endl;
  std::cout << "METRIC_TYPE: " << cfg.metric_type << std::endl;
  std::cout << "Index path: " << cfg.index_path << std::endl;
  return cfg;
}
}  // namespace

int main(int argc, char** argv) {
  auto cfg = parse_config(argc, argv);
  // dataset
  int n = cfg.data_n;
  int d = cfg.data_dim;
  // configuration
  int nlist = 0;
  int default_bbs = 32;
  int use_multithread = 0;

  // search
  int k = cfg.search_k;

  // load data
  float* data = nullptr;
  data = load_data(cfg.data_train_path.c_str(), d, n);

  faiss::MetricType metric =
      (cfg.metric_type == 1) ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;

  printf("Memory usage before initialization: %ld\n",
         faiss::getCurrentRSS() / 1024 / 1024);

  auto index_start = std::chrono::high_resolution_clock::now();
  std::unique_ptr<faiss::HakesIndex> index;

  // check if the index directory exists
  if (!std::filesystem::exists(cfg.index_path)) {
    throw std::runtime_error("Index directory not found");
    return -1;
  }

  // load index
  index.reset(new faiss::HakesIndex());
  printf("Loading index from %s\n", cfg.index_path.c_str());
  if (faiss::MetricType::METRIC_L2 == metric) {
    index->use_ivf_sq_ = false;
  } else {
    index->use_ivf_sq_ = true;
  }
  index->Initialize(cfg.index_path.c_str(), 0);

  // use power-of-3 assignment
  index->base_index_->use_balanced_assign_ = false;
  index->base_index_->balance_k_ = 3;

  std::cout << index->to_string() << std::endl;

  auto xids = std::make_unique<faiss::idx_t[]>(n);
  for (int i = 0; i < n; ++i) {
    xids[i] = i;
  }
  std::unique_ptr<faiss::idx_t[]> assign =
      std::make_unique<faiss::idx_t[]>(n);
  int vecs_t_d;
  std::unique_ptr<float[]> vecs_t;

  // for 10M data need to load by batches
  auto batch_size = 100000;
  for (int i = 0; i < n; i += batch_size) {
    index->AddWithIds(std::min(batch_size, n - i), d, data + i * d,
                      xids.get() + i, assign.get() + i, &vecs_t_d, &vecs_t);

    std::cout << "Inserted " << i << " vector" << std::endl;
  }

  // if update index exists use its parameters instead
  // note that if the uindex.bin is already placed under the index_path,
  // then parameters in the uindex.bin is already loaded.
  if (!cfg.update_index.empty()) {
    faiss::HakesIndex update_index;
    printf("Loading update index from %s\n", cfg.update_index.c_str());
    update_index.Initialize(cfg.update_index.c_str());
    index->UpdateIndex(&update_index);
  }

  printf("Memory usage after initialization: %ld\n",
         faiss::getCurrentRSS() / 1024 / 1024);

  auto index_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> index_diff = index_end - index_start;
  std::cout << "index load time (s): " << index_diff.count() << std::endl;

  omp_set_dynamic(0);
  omp_set_num_threads(0);

  auto base_index =
      dynamic_cast<faiss::IndexIVFPQFastScanL*>(index->base_index_.get());
  if (base_index == nullptr) {
    throw std::runtime_error("Base index is not IVFPQFastScanL");
  }
  nlist = base_index->quantizer->ntotal;

  auto sample_prepare_start_time = std::chrono::high_resolution_clock::now();
  // sample 100k queries topK with 100 neighbors with 1/10 lists
  int sampled_count = 100 * 1000;
  int nprobe = nlist / 10;
  int k_factor = 10;
  faiss::idx_t k_base = k * k_factor;
  if (k_base < 100) {
    throw std::runtime_error("k_base should be at least 100");
  }
  printf(
      "sample queries results for 100 sampled_count: %d, nprobe: %d "
      "k_factor: %d\n",
      sampled_count, nprobe, k_factor);
  faiss::HakesSearchParams params;
  params.nprobe = nprobe;
  params.k = k;
  params.k_factor = k_factor;
  params.metric_type = metric;

  std::unique_ptr<faiss::idx_t[]> candidates;
  std::unique_ptr<float[]> distances;
  // prepare the query
  auto sampled_query = std::make_unique<float[]>(sampled_count * d);
  // int step = nq / sampled_count;
  int step = n / sampled_count;
  for (int i = 0; i < sampled_count; ++i) {
    std::memcpy(sampled_query.get() + i * d, data + i * step * d,
                d * sizeof(float));
  }

  index->Search(sampled_count, d, sampled_query.get(), params, &distances,
                &candidates);
  std::unique_ptr<faiss::idx_t[]> k_base_count =
      std::make_unique<faiss::idx_t[]>(sampled_count);
  for (int i = 0; i < sampled_count; ++i) {
    k_base_count[i] = k_base;
  }
  index->Rerank(sampled_count, d, sampled_query.get(), k, k_base_count.get(),
                candidates.get(), distances.get(), &distances, &candidates);
  auto nn_file_name = cfg.index_path + "/sample100NN_" +
                      std::to_string(sampled_count) + ".bin";
  auto query_file_name = cfg.index_path + "/sampleQuery_" +
                          std::to_string(sampled_count) + ".bin";
  std::ofstream out(nn_file_name, std::ofstream::out | std::ofstream::binary);
  for (int i = 0; i < sampled_count; ++i) {
    for (int j = 0; j < k; ++j) {
      out.write((char*)(candidates.get() + i * k + j), sizeof(int32_t));
    }
  }
  out.close();
  // store the sampled queries in binary
  out.open(query_file_name, std::ofstream::out | std::ofstream::binary);
  out.write((char*)sampled_query.get(), sampled_count * d * sizeof(float));
  out.close();
  printf("saved sample queries and 100NN results (sampled_count: %d)\n",
          sampled_count);

  auto sample_prepare_end_time = std::chrono::high_resolution_clock::now();
  printf("sample prepare time (s): %f\n",
          std::chrono::duration<double>(sample_prepare_end_time -
                                        sample_prepare_start_time)
              .count());

  // clean up
  delete[] data;
  return 0;
}
