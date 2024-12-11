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
  size_t data_num_query;
  size_t data_dim;
  size_t search_k = 20;  // top_k
  size_t data_groundtruth_len;
  std::string data_train_path;
  std::string data_query_path;
  std::string data_groundtruth_path;
  // index params
  bool load_index = false;
  int test_search_only = false;
  size_t num_client = 1;
  float write_ratio = 0.1;
  int batch_size = 1;
  int nprobe = 100;
  int k_factor = 50;

  std::string index_path = "./index";
  std::string update_index = "";
  size_t report_interval = 10000;
};

Config parse_config(int argc, char** argv) {
  Config cfg;
  if (argc < 16) {
    std::cout << "Usage: " << argv[0]
              << "DATA_N DATA_NUM_QUERY DATA_DIM SEARCH_K DATA_GROUNDTRUTH_LEN "
                 "DATA_TRAIN_PATH DATA_QUERY_PATH DATA_GROUNDTRUTH_PATH "
                 "LOAD_INDEX TEST_SEARCH_ONLY NUM_CLIENT WRITE_RATIO "
                 "BATCH_SIZE NPROBE K_FACTOR INDEX_PATH [UPDATE INDEX PARAMS]"
              << std::endl;
    exit(-1);
  }

  cfg.data_n = std::stoul(argv[1]);
  cfg.data_num_query = std::stoul(argv[2]);
  cfg.data_dim = std::stoul(argv[3]);
  cfg.search_k = std::stoul(argv[4]);
  cfg.data_groundtruth_len = std::stoul(argv[5]);
  cfg.data_train_path = argv[6];
  cfg.data_query_path = argv[7];
  cfg.data_groundtruth_path = argv[8];
  cfg.load_index = std::stoul(argv[9]);
  cfg.test_search_only = std::stoul(argv[10]);
  cfg.num_client = std::stoul(argv[11]);
  cfg.write_ratio = std::stof(argv[12]);
  cfg.batch_size = std::stoul(
      argv[13]);  // when batch_size is set to > 1, write ratio will be set to 0

  if (cfg.batch_size > 1) {
    cfg.write_ratio = 0;
  }
  cfg.nprobe = std::stoul(argv[14]);
  cfg.k_factor = std::stoul(argv[15]);
  if (argc > 16) {
    cfg.index_path = argv[16];
  }
  if (argc > 17) {
    cfg.update_index = argv[17];
  }

  // print summary
  std::cout << "DATA_N: " << cfg.data_n << std::endl;
  std::cout << "DATA_NUM_QUERY: " << cfg.data_num_query << std::endl;
  std::cout << "DATA_DIM: " << cfg.data_dim << std::endl;
  std::cout << "SEARCH_K: " << cfg.search_k << std::endl;
  std::cout << "DATA_GROUNDTRUTH_LEN: " << cfg.data_groundtruth_len
            << std::endl;
  std::cout << "DATA_TRAIN_PATH: " << cfg.data_train_path << std::endl;
  std::cout << "DATA_QUERY_PATH: " << cfg.data_query_path << std::endl;
  std::cout << "DATA_GROUNDTRUTH_PATH: " << cfg.data_groundtruth_path
            << std::endl;
  std::cout << "LOAD_INDEX: " << cfg.load_index << std::endl;
  std::cout << "TEST_SEARCH_ONLY: " << cfg.test_search_only << std::endl;
  std::cout << "NUM_CLIENT: " << cfg.num_client << std::endl;
  std::cout << "WRITE_RATIO: " << cfg.write_ratio << std::endl;
  std::cout << "BATCH_SIZE: " << cfg.batch_size << std::endl;
  std::cout << "NPROBE: " << cfg.nprobe << std::endl;
  std::cout << "K_FACTOR: " << cfg.k_factor << std::endl;
  std::cout << "Index path: " << cfg.index_path << std::endl;
  if (!cfg.update_index.empty()) {
    std::cout << "Update index params: " << cfg.update_index << std::endl;
  }
  return cfg;
}

std::mutex final_print_mutex;

void run_client(faiss::HakesIndex* index, float* query, int nq, int k, int d,
                int id, int nd, const Config& cfg,
                faiss::HakesSearchParams* search_params = nullptr) {
  thread_launched++;
  while (thread_launched < cfg.num_client) {
    std::this_thread::yield();
  }
  // std::ofstream out("Hakes-output" + std::to_string(id) + ".out",
  //                   std::ofstream::out | std::ofstream::app);
  // int batch_size = 20;
  int batch_size = cfg.batch_size;
  // bool write_client = id < WRITE_CLIENT;
  auto start = std::chrono::high_resolution_clock::now();
  double total_time = 0;
  // generate read write sequence
  std::mt19937 gen(id);
  std::uniform_real_distribution<> dis(0, 1);
  std::vector<bool> write_sequence(nq);
  for (int i = 0; i < nq; ++i) {
    write_sequence[i] = dis(gen) < cfg.write_ratio;
  }

  // stats variables
  int write_op_executed = 0;
  int read_op_executed = 0;
  int executed = 0;
  std::vector<float> write_lats;
  write_lats.reserve(nq);
  std::vector<float> read_lats;
  read_lats.reserve(nq);

  int k_base = k * search_params->k_factor;
  float base_search_time_sum = 0.0;
  float rerank_search_time_sum = 0.0;

  for (int i = 0; i < nq / batch_size; ++i) {
    auto per_query_start = std::chrono::high_resolution_clock::now();
    if (write_sequence[i]) {
      faiss::idx_t xids = i + nd;
      std::unique_ptr<faiss::idx_t[]> assign =
          std::make_unique<faiss::idx_t[]>(1);
      int vecs_t_d;
      std::unique_ptr<float[]> vecs_t;
      index->AddWithIds(1, d, query + i * d, &xids, assign.get(), &vecs_t_d,
                        &vecs_t);
      write_op_executed++;
    } else {
      std::unique_ptr<faiss::idx_t[]> candidates;
      std::unique_ptr<float[]> distances;
      auto base_start = std::chrono::high_resolution_clock::now();
      index->Search(batch_size, d, query + i * batch_size * d, *search_params,
                    &distances, &candidates);
      std::unique_ptr<faiss::idx_t[]> k_base_count =
          std::make_unique<faiss::idx_t[]>(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        k_base_count[i] = k_base;
      }
      auto base_end = std::chrono::high_resolution_clock::now();
      index->Rerank(batch_size, d, query + i * batch_size * d, k,
                    k_base_count.get(), candidates.get(), distances.get(),
                    &distances, &candidates);
      auto rerank_end = std::chrono::high_resolution_clock::now();

      base_search_time_sum +=
          std::chrono::duration<double>(base_end - base_start).count();
      rerank_search_time_sum +=
          std::chrono::duration<double>(rerank_end - base_end).count();

      read_op_executed += batch_size;
      // // save the search result
      // for (int j = 0; j < k; ++j) {
      //   out << candidates[j] << " ";
      // }
    }
    executed += batch_size;
    // out << "\n";
    auto per_query_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> per_query_diff =
        per_query_end - per_query_start;
    total_time += per_query_diff.count();
    // out << "client " << id << " query " << i
    //     << " time (us): " << per_query_diff.count() * 1000000 << "\n";
    if (write_sequence[i]) {
      write_lats.emplace_back(per_query_diff.count());
    } else {
      read_lats.emplace_back(per_query_diff.count());
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  // print stats
  float sum_write_time = 0;
  for (int i = 0; i < write_lats.size(); ++i) {
    sum_write_time += write_lats[i];
  }
  float sum_read_time = 0;
  for (int i = 0; i < read_lats.size(); ++i) {
    sum_read_time += read_lats[i];
  }
  std::sort(write_lats.begin(), write_lats.end());
  std::sort(read_lats.begin(), read_lats.end());
  // out << "client " << id << " stats: \n";
  auto duration = diff.count();

  std::lock_guard<std::mutex> lock(final_print_mutex);
  std::cout << "client " << id << " stats: \n";
  std::cout << "TOTAL - Takes (s): " << duration << " Count: " << executed
            << " OPS: " << float(executed) / duration
            << " Avg(ms): " << duration * 1000 / executed << "\n";
  if (read_op_executed > 0) {
    std::cout << "SEARCH - Takes (s): " << duration
              << " Count: " << read_op_executed
              << " BATCH_COUNT: " << read_lats.size()
              << " OPS: " << float(read_op_executed) / duration
              << " Avg(ms): " << sum_read_time * 1000 / read_lats.size()
              << " 50th(ms): " << read_lats[read_lats.size() * 50 / 100] * 1000
              << " 95th(ms): " << read_lats[read_lats.size() * 95 / 100] * 1000
              << " 99th(ms): " << read_lats[read_lats.size() * 99 / 100] * 1000
              << "\n";
  }
  if (write_op_executed > 0) {
    std::cout << "ADD - Takes (s): " << duration
              << " Count: " << write_op_executed
              << " BATCH_COUNT: " << write_lats.size()
              << " OPS: " << float(write_op_executed) / duration
              << " Avg(ms): " << sum_write_time * 1000 / write_lats.size()
              << " 50th(ms): "
              << write_lats[write_lats.size() * 50 / 100] * 1000
              << " 95th(ms): "
              << write_lats[write_lats.size() * 95 / 100] * 1000
              << " 99th(ms): "
              << write_lats[write_lats.size() * 99 / 100] * 1000 << "\n";
  }

  std::cout << "base ratio: "
            << base_search_time_sum /
                   (base_search_time_sum + rerank_search_time_sum)
            << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  auto cfg = parse_config(argc, argv);
  // dataset
  int n = cfg.data_n;
  int d = cfg.data_dim;
  int nq = cfg.data_num_query;
  int gt_len = cfg.data_groundtruth_len;
  // configuration
  int nlist = 0;
  int default_bbs = 32;
  // bool use_multithread = false;
  // bool use_multithread = true;
  // int use_multithread = 32;
  int use_multithread = 0;

  // search
  int k = cfg.search_k;

  // load data
  float* data = nullptr;
  data = load_data(cfg.data_train_path.c_str(), d, n);
  float* query = nullptr;
  query = load_data(cfg.data_query_path.c_str(), d, nq);
  int* gt = nullptr;
  gt = load_groundtruth(cfg.data_groundtruth_path.c_str(), gt_len, nq);

  faiss::MetricType metric = faiss::METRIC_INNER_PRODUCT;

  printf("Memory usage before initialization: %ld\n",
         faiss::getCurrentRSS() / 1024 / 1024);

  auto index_start = std::chrono::high_resolution_clock::now();
  std::unique_ptr<faiss::HakesIndex> index;
  if (!cfg.load_index) {
    throw std::runtime_error("Index need to be loaded");
  } else {
    // check if the index directory exists
    if (!std::filesystem::exists(cfg.index_path)) {
      throw std::runtime_error("Index directory not found");
      return -1;
    }

    // load index
    index.reset(new faiss::HakesIndex());
    printf("Loading index from %s\n", cfg.index_path.c_str());
    index->use_ivf_sq_ = true;
    index->Initialize(cfg.index_path.c_str());

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
    if (!cfg.update_index.empty()) {
      faiss::HakesIndex update_index;
      printf("Loading update index from %s\n", cfg.update_index.c_str());
      update_index.Initialize(cfg.update_index.c_str());
      index->UpdateIndex(update_index);
    }
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

  {
    // set termination parameters
    index->base_index_->use_early_termination_ = true;
    index->base_index_->et_params.beta = 200;
    index->base_index_->et_params.ce = 30;
  }

  if (cfg.test_search_only == 1) {
    // search
    auto nprobe_list = std::vector<int>{1,   5,   10,  50, 100,
                                        200, 300, 400, 500};  // for 1M dataset
    // auto nprobe_list =
    //     std::vector<int>{1,   5,   10,  50,  100,  200, 300,
    //                      400, 500, 600, 800, 1000, 1200};  // for 10M dataset
    auto k_factor_list =
        std::vector<int>{1, 10, 20, 50, 100, 200, 300, 400, 500, 600};

    float recall_early_stop = 0.9999;
    // float recall_early_stop = 0.995;

    float last_recall = 0;

    for (auto nprobe : nprobe_list) {
      for (auto k_factor : k_factor_list) {
        if (nprobe > nlist) {
          continue;
        }
        auto k_base = k * k_factor;

        faiss::HakesSearchParams params;
        params.nprobe = nprobe;
        params.k = k;
        params.k_factor = k_factor;
        params.metric_type = metric;

        auto result = std::make_unique<faiss::idx_t[]>(k * nq);

        auto start = std::chrono::high_resolution_clock::now();
        if (use_multithread == 1) {
          std::unique_ptr<faiss::idx_t[]> candidates;
          std::unique_ptr<float[]> distances;
          index->Search(nq, d, query, params, &distances, &candidates);

          std::unique_ptr<faiss::idx_t[]> k_base_count =
              std::make_unique<faiss::idx_t[]>(nq);
          for (int i = 0; i < nq; ++i) {
            k_base_count[i] = k_base;
          }
          index->Rerank(nq, d, query, k, k_base_count.get(), candidates.get(),
                        distances.get(), &distances, &candidates);
          std::memcpy(result.get(), candidates.get(),
                      k * nq * sizeof(faiss::idx_t));
        } else if (use_multithread == 0) {
          for (int i = 0; i < nq; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();
            std::unique_ptr<faiss::idx_t[]> candidates;
            std::unique_ptr<float[]> distances;
            index->Search(1, d, query + i * d, params, &distances, &candidates);
            auto search_end_time = std::chrono::high_resolution_clock::now();
            std::unique_ptr<faiss::idx_t[]> k_base_count =
                std::make_unique<faiss::idx_t[]>(1);
            k_base_count[0] = k_base;
            index->Rerank(1, d, query + i * d, k, k_base_count.get(),
                          candidates.get(), distances.get(), &distances,
                          &candidates);
            std::memcpy(result.get() + i * k, candidates.get(),
                        k * sizeof(faiss::idx_t));
            auto end_time = std::chrono::high_resolution_clock::now();
          }
        } else {
          // multiple thread to search concurrently
          std::vector<std::thread> clients;
          // distribute the request evenly
          for (int i = 0; i < use_multithread; i++) {
            clients.emplace_back(
                std::thread([i, &index, &query, &params, &result, k, k_base, d,
                             nq, use_multithread]() {
                  for (int j = i; j < nq; j += use_multithread) {
                    std::unique_ptr<faiss::idx_t[]> candidates;
                    std::unique_ptr<float[]> distances;
                    index->Search(1, d, query + j * d, params, &distances,
                                  &candidates);
                    std::unique_ptr<faiss::idx_t[]> k_base_count =
                        std::make_unique<faiss::idx_t[]>(1);
                    k_base_count[0] = k_base;
                    index->Rerank(1, d, query + j * d, k, k_base_count.get(),
                                  candidates.get(), distances.get(), &distances,
                                  &candidates);
                    std::memcpy(result.get() + j * k, candidates.get(),
                                k * sizeof(faiss::idx_t));
                  }
                }

                            ));
          }
          // join the threads
          for (auto& client : clients) {
            client.join();
          }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        // evaluate
        int correct = 0;
        for (int i = 0; i < nq; ++i) {
          // prepare the groundtruth set
          std::unordered_set<int> gts(k);
          for (int j = 0; j < k; ++j) {
            gts.insert(gt[i * gt_len + j]);
          }

          for (int j = 0; j < k; ++j) {
            if (gts.find(result[i * k + j]) != gts.end()) {
              correct++;
            }
          }
        }
        // print the recall
        std::cout << "nprobe: " << nprobe << " k_factor: " << k_factor
                  << " search time (s): " << diff.count()
                  << " recall: " << correct / (float)(nq * k) << std::endl;
        last_recall = correct / (float)(nq * k);
      }
      if (last_recall > recall_early_stop) {
        break;
      }
    }
  } else if (cfg.test_search_only == 2) {
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
  } else {
    assert(cfg.test_search_only == 0);
    // reserve the space in flat beforehand
    index->refine_index_->reserve(n + nq);
    std::cout << "start workload" << std::endl;
    // search and insert
    faiss::HakesSearchParams search_params;
    search_params.nprobe = cfg.nprobe;
    search_params.k = cfg.search_k;
    search_params.k_factor = cfg.k_factor;
    std::cout << "nprobe: " << search_params.nprobe << " k: " << search_params.k
              << " k_factor: " << search_params.k_factor << std::endl;
    std::vector<std::thread> clients;
    for (int i = 0; i < cfg.num_client; ++i) {
      auto num_per_client = nq / cfg.num_client;
      auto start = i * nq / cfg.num_client;
      clients.emplace_back(std::thread(run_client, index.get(),
                                       query + start * d, num_per_client, k, d,
                                       i, n, cfg, &search_params));
    }
    for (auto& client : clients) {
      client.join();
    }
  }

  // clean up
  delete[] data;
  delete[] query;
  delete[] gt;
  return 0;
}
