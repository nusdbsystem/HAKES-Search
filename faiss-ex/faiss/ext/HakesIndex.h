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

#ifndef HAKES_Hakes_INDEX_H_
#define HAKES_Hakes_INDEX_H_

#include <faiss/VectorTransform.h>
#include <faiss/ext/IdMap.h>
#include <faiss/ext/IndexFlatL.h>
#include <faiss/ext/IndexIVFPQFastScanL.h>

namespace faiss {

struct HakesSearchParams {
  int nprobe;
  int k;
  int k_factor;
  faiss::MetricType metric_type;
};

struct HAKESStats {
  uint64_t query_count = 0;
  double accu_vt_time = 0;
  double accu_pa_time = 0;
  double accu_qs_time = 0;
  double accu_rerank_time = 0;
};

class HakesIndex {
 public:
  HakesIndex() = default;
  ~HakesIndex() {
    for (auto vt : vts_) {
      delete vt;
    }
    for (auto vt : ivf_vts_) {
      delete vt;
    }
    for (auto vt : q_vts_) {
      delete vt;
    }
    for (auto vt : q_ivf_vts_) {
      delete vt;
    }
    if (use_ivf_sq_) {
      delete cq_;
    }
    if (q_cq_) {
      delete q_cq_;
    }
  }

  // delete copy constructors and assignment operators
  HakesIndex(const HakesIndex&) = delete;
  HakesIndex& operator=(const HakesIndex&) = delete;
  // delete move constructors and assignment operators
  HakesIndex(HakesIndex&&) = delete;
  HakesIndex& operator=(HakesIndex&&) = delete;

  bool Initialize(const std::string& path, bool keep_pa = false);

  void UpdateIndex(HakesIndex& other);

  bool AddWithIds(int n, int d, const float* vecs, const faiss::idx_t* ids) {
    printf(
        "HakesIndex::AddWithIds need to return results of base index "
        "assignment\n");
    return false;
  }

  // it is assumed that receiving engine shall store the full vecs of all
  // inputs.
  bool AddWithIds(int n, int d, const float* vecs, const faiss::idx_t* ids,
                  faiss::idx_t* assign, int* vecs_t_d,
                  std::unique_ptr<float[]>* vecs_t);

  bool AddBasePreassigned(int n, int d, const float* vecs,
                          const faiss::idx_t* ids, const faiss::idx_t* assign);

  bool AddToRefine(int n, int d, const float* vecs, const faiss::idx_t* xids,
                   const faiss::idx_t* assign);

  bool Search(int n, int d, const float* query, const HakesSearchParams& params,
              std::unique_ptr<float[]>* distances,
              std::unique_ptr<faiss::idx_t[]>* labels);

  bool SearchAndPA(int n, int d, const float* query,
                   const HakesSearchParams& params,
                   std::unique_ptr<float[]>* distances,
                   std::unique_ptr<faiss::idx_t[]>* labels,
                   std::unique_ptr<faiss::idx_t[]>* pa);

  bool Rerank(int n, int d, const float* query, int k,
              faiss::idx_t* k_base_count, faiss::idx_t* base_labels,
              float* base_distances, std::unique_ptr<float[]>* distances,
              std::unique_ptr<faiss::idx_t[]>* labels);

  inline bool has_ivf_vts() const { return ivf_vts_.size() > 0; }

  void Reserve(faiss::idx_t n);

  bool Checkpoint(const std::string& checkpoint_path);

  // single file check out
  bool Checkpoint(IOWriter* f);

  // external synchronization needed to not call this function concurrently with
  // UpdateParams
  std::string GetParams() const;

  // external synchronization needed to not call this function concurrently with
  // other operations
  bool UpdateParams(const std::string& params);

  std::string to_string() const;

  inline HAKESStats GetStats() const {
    std::shared_lock lock(stats_mu_);
    return stats_;
  }

  inline void ResetStats() {
    std::lock_guard lock(stats_mu_);
    stats_ = HAKESStats();
  }

 public:
  std::string index_path_;
  bool share_vt_;
  bool use_ivf_sq_ = false;
  std::vector<faiss::VectorTransform*> vts_;
  std::vector<faiss::VectorTransform*> ivf_vts_;
  bool has_q_index_ = false;
  std::vector<faiss::VectorTransform*> q_vts_;
  std::vector<faiss::VectorTransform*> q_ivf_vts_;
  faiss::Index* cq_;
  faiss::Index* q_cq_ = nullptr;
  std::unique_ptr<faiss::IndexIVFPQFastScanL> base_index_;
  std::shared_mutex mapping_mu_;
  std::unique_ptr<faiss::IDMap> mapping_;
  std::unique_ptr<faiss::IndexFlatL> refine_index_;

  bool collect_stats_ = false;
  HAKESStats stats_;
  mutable std::shared_mutex stats_mu_;

  bool keep_pa_;
  std::unordered_map<faiss::idx_t, faiss::idx_t> pa_mapping_;
};

}  // namespace faiss

#endif  // HAKES_Hakes_INDEX_H_
