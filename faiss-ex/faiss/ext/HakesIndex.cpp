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

#include <faiss/ext/HakesIndex.h>
#include <faiss/ext/IndexIVFFastScanL.h>
#include <faiss/ext/IndexScalarQuantizerL.h>
#include <faiss/ext/index_io_ext.h>
#include <faiss/ext/io_ext.h>
#include <faiss/ext/utils.h>
#include <faiss/impl/FaissAssert.h>

#include <filesystem>
#include <vector>

#define FINDEX_NAME "findex.bin"
#define RINDEX_NAME "rindex.bin"
#define UINDEX_NAME "uindex.bin"
#define PAMAP_NAME "pamap.bin"

namespace faiss {

bool HakesIndex::Initialize(const std::string& path, int mode, bool keep_pa) {
  std::string findex_path = path + "/" + FINDEX_NAME;
  std::string rindex_path = path + "/" + RINDEX_NAME;
  std::string uindex_path = path + "/" + UINDEX_NAME;

  std::unique_ptr<FileIOReader> ff =
      IsFileExist(findex_path)
          ? std::make_unique<FileIOReader>(findex_path.c_str())
          : nullptr;
  std::unique_ptr<FileIOReader> rf =
      IsFileExist(rindex_path)
          ? std::make_unique<FileIOReader>(rindex_path.c_str())
          : nullptr;
  std::unique_ptr<FileIOReader> uf =
      IsFileExist(uindex_path)
          ? std::make_unique<FileIOReader>(uindex_path.c_str())
          : nullptr;

  // output memory before loading index
  printf("Memory usage before loading index: %ld\n",
         getCurrentRSS() / 1024 / 1024);

  bool success = load_hakes_index(ff.get(), rf.get(), this, mode);
  if (!success) {
    fprintf(stderr, "Failed to load index from %s\n", path.c_str());
    return false;
  }

  if (mode == 2) {
    assert(base_index_ == nullptr);
    assert(mapping_);
    assert(refine_index_);
    printf("HAKESIndex Refine Index Only Completed\n");
    return success;
  }

  if (uf != nullptr) {
    // load search index parameters
    HakesIndex update_index;
    // success = load_hakes_index(uf.get(), nullptr, &update_index, 1);
    success = load_hakes_params(uf.get(), &update_index);
    this->UpdateIndex(&update_index);
  } else if (use_ivf_sq_) {
    // no query index provided, apply sq to the ivf centroids
    auto tmp = new IndexScalarQuantizerL(
        base_index_->d, ScalarQuantizer::QuantizerType::QT_8bit_direct,
        faiss::MetricType::METRIC_L2);
    int nlist = base_index_->quantizer->get_ntotal();
    auto scaled_codes = std::vector<float>(nlist * base_index_->d);
    const float* to_scale =
        (const float*)static_cast<IndexFlatL*>(base_index_->quantizer)
            ->codes.data();
    for (int i = 0; i < nlist * base_index_->d; i++) {
      scaled_codes[i] = to_scale[i] * 127 + 128;
    }
    tmp->add(nlist, scaled_codes.data());
    cq_ = tmp;
  } else {
    cq_ = base_index_->quantizer;
  }
  base_index_->use_balanced_assign_ = false;
  if (!success) {
    fprintf(stderr, "Failed to load index from %s\n", path.c_str());
    return false;
  }

  {
    if (keep_pa) {
      // load pa mapping
      std::string pa_mapping_path = path + "/" + PAMAP_NAME;
      if (std::filesystem::exists(pa_mapping_path)) {
        std::unique_ptr<FileIOReader> pa_mapping_file =
            std::make_unique<FileIOReader>(pa_mapping_path.c_str());
        load_pa_map(pa_mapping_file.get(), this);
        printf("Loaded pa mapping from %s (entries: %ld)\n",
               pa_mapping_path.c_str(), pa_mapping_.size());
      } else {
        printf("No pa mapping found in %s\n", pa_mapping_path.c_str());
      }
      keep_pa_ = true;
    } else {
      keep_pa_ = false;
    }
  }

  if (refine_index_ && mapping_->size() != refine_index_->ntotal) {
    throw std::runtime_error("mapping size not equal to refine index size");
  }

  printf("loaded index: %s\n", this->to_string().c_str());

  printf("Memory usage after initialization: %ld\n",
         getCurrentRSS() / 1024 / 1024);

  return true;
}

void HakesIndex::UpdateIndex(const HakesIndex* update_index) {
  if (base_index_ == nullptr) {
    printf("filter index is not initialized, so cannot perform UpdateIndex\n");
    return;
  }
  assert(update_index->base_index_->metric_type == base_index_->metric_type);
  assert(update_index->cq_->ntotal == update_index->cq_->ntotal);
  assert(update_index->vts_.back()->d_out == base_index_->d);
  assert(update_index->base_index_->by_residual == base_index_->by_residual);
  assert(update_index->base_index_->code_size == base_index_->code_size);
  assert(update_index->base_index_->ksub == base_index_->ksub);
  assert(update_index->base_index_->nbits == base_index_->nbits);
  assert(update_index->base_index_->M == base_index_->M);
  assert(update_index->base_index_->M2 == base_index_->M2);
  assert(update_index->base_index_->implem == base_index_->implem);
  assert(update_index->base_index_->qbs2 == base_index_->qbs2);
  assert(update_index->base_index_->bbs == base_index_->bbs);

  // new vts
  std::vector<VectorTransform*> new_vts_;
  new_vts_.reserve(update_index->vts_.size());
  for (auto vt : update_index->vts_) {
    auto lt = dynamic_cast<LinearTransform*>(vt);
    LinearTransform* new_vt = new LinearTransform(vt->d_in, vt->d_out);
    new_vt->A = lt->A;
    new_vt->b = lt->b;
    new_vt->have_bias = lt->have_bias;
    new_vt->is_trained = lt->is_trained;
    new_vts_.push_back(new_vt);
  }

  // new base index pq
  assert(update_index->base_index_->pq.M == base_index_->pq.M);
  assert(update_index->base_index_->pq.nbits == base_index_->pq.nbits);

  // default behavior is to always install the updated parameters as search
  // index parameters
  has_q_index_ = true;

  if (has_q_index_) {
    for (auto vt : q_vts_) {
      delete vt;
    }
    q_vts_ = new_vts_;
  } else {
    for (auto vt : vts_) {
      delete vt;
    }
    vts_ = new_vts_;
  }

  // new quantizer
  IndexFlatL* new_quantizer =
      new IndexFlatL(update_index->base_index_->quantizer->d,
                     update_index->base_index_->quantizer->metric_type);
  new_quantizer->ntotal = update_index->base_index_->quantizer->ntotal;
  new_quantizer->is_trained = update_index->base_index_->quantizer->is_trained;
  new_quantizer->codes =
      dynamic_cast<IndexFlatL*>(update_index->base_index_->quantizer)->codes;

  delete q_quantizer_;
  q_quantizer_ = new_quantizer;

  if (use_ivf_sq_) {
    assert(update_index->cq_);
    printf("apply sq to the update index ivf quantizer\n");
    auto tmp = new IndexScalarQuantizerL(
        update_index->base_index_->d,
        ScalarQuantizer::QuantizerType::QT_8bit_direct, METRIC_L2);
    int nlist = update_index->base_index_->quantizer->get_ntotal();
    auto scaled_codes =
        std::vector<float>(nlist * update_index->base_index_->d);
    const float* to_scale = (const float*)static_cast<IndexFlatL*>(
                                update_index->base_index_->quantizer)
                                ->codes.data();
    for (int i = 0; i < nlist * update_index->base_index_->d; i++) {
      scaled_codes[i] = to_scale[i] * 127 + 128;
    }
    tmp->add(nlist, scaled_codes.data());
    delete q_cq_;
    q_cq_ = tmp;
  } else {
    q_cq_ = q_quantizer_;
  }

  assert(base_index_->pq.d == update_index->base_index_->pq.d);
  assert(base_index_->pq.M == update_index->base_index_->pq.M);
  assert(base_index_->pq.nbits == update_index->base_index_->pq.nbits);

  base_index_->has_q_pq = true;
  base_index_->q_pq = update_index->base_index_->pq;
}

bool HakesIndex::AddWithIds(int n, int d, const float* vecs,
                            const faiss::idx_t* xids, faiss::idx_t* assign,
                            int* vecs_t_d,
                            std::unique_ptr<float[]>* transformed_vecs) {
  auto start = std::chrono::high_resolution_clock::now();

  if (refine_index_) {
    // serve as refine_worker
    std::unique_lock lock(mapping_mu_);
    mapping_->add_ids(n, xids);
    refine_index_->add(n, vecs);
  }

  auto refine_add_end = std::chrono::high_resolution_clock::now();

  if (base_index_ == nullptr) {
    // serve only as refine worker
    return true;
  }

  // serve as index worker
  if (vts_.empty()) {
    base_index_->add_with_ids(n, vecs, xids, false, assign);
    return true;
  }

  // get assignment with cq first
  std::vector<float> vecs_t(n * d);
  bool assigned = false;

  vecs_t.resize(n * d);
  std::memcpy(vecs_t.data(), vecs, n * d * sizeof(float));
  std::vector<float> tmp;
  for (auto vt : vts_) {
    tmp.resize(n * vt->d_out);
    std::memset(tmp.data(), 0, tmp.size() * sizeof(float));
    vt->apply_noalloc(n, vecs_t.data(), tmp.data());
    vecs_t.resize(tmp.size());
    std::memcpy(vecs_t.data(), tmp.data(), tmp.size() * sizeof(float));
  }

  base_index_->get_add_assign(n, vecs_t.data(), assign);
  assigned = true;

  // return the transformed vecs.
  *vecs_t_d = vecs_t.size() / n;
  transformed_vecs->reset(new float[vecs_t.size()]);
  std::memcpy(transformed_vecs->get(), vecs_t.data(),
              vecs_t.size() * sizeof(float));

  auto vt_add_end = std::chrono::high_resolution_clock::now();

  if (keep_pa_) {
    std::unique_lock lock(mapping_mu_);
    for (int i = 0; i < n; i++) {
      pa_mapping_.emplace(xids[i], assign[i]);
    }
  }
  auto pa_add_end = std::chrono::high_resolution_clock::now();

  assert(vecs_t.size() == n * base_index_->d);
  // base_index_->add_with_ids(n, vecs_t.data(), xids, false, assign);
  base_index_->add_with_ids(n, vecs_t.data(), xids, assigned, assign);

  auto base_add_end = std::chrono::high_resolution_clock::now();
  return true;
}

bool HakesIndex::AddToRefine(int n, int d, const float* vecs,
                             const faiss::idx_t* xids,
                             const faiss::idx_t* assign) {
  assert(refine_index_);
  {
    std::unique_lock lock(mapping_mu_);
    mapping_->add_ids(n, xids);
    refine_index_->add(n, vecs);
  }

  return true;
}

bool HakesIndex::AddBasePreassigned(int n, int d, const float* vecs,
                                    const faiss::idx_t* xids,
                                    const faiss::idx_t* assign) {
  assert(base_index_);
  assert(base_index_->d == d);

  if (keep_pa_) {
    std::unique_lock lock(mapping_mu_);
    for (int i = 0; i < n; i++) {
      pa_mapping_.emplace(xids[i], assign[i]);
    }
  }

  base_index_->add_with_ids(n, vecs, xids, true,
                            const_cast<faiss::idx_t*>(assign));
  return true;
}

bool HakesIndex::Search(int n, int d, const float* query,
                        const HakesSearchParams& params,
                        std::unique_ptr<float[]>* distances,
                        std::unique_ptr<faiss::idx_t[]>* labels) {
  if (!distances || !labels) {
    throw std::runtime_error("distances and labels must not be nullptr");
  }

  if (n <= 0 || params.k <= 0 || params.k_factor <= 0 || params.nprobe <= 0) {
    printf(
        "n, k, k_factor and nprobe must > 0: n %d, k: %d, k_factor: %d, "
        "nprobe: %d\n",
        n, params.k, params.k_factor, params.nprobe);
    return false;
  }

  auto opq_vt_start = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> opq_alloc_time;
  std::chrono::duration<double> opq_apply_time;

  auto q_vts = (has_q_index_) ? q_vts_ : vts_;
  std::vector<float> query_t(n * d);
  std::memcpy(query_t.data(), query, n * d * sizeof(float));
  {
    std::vector<float> tmp;
    for (auto vt : q_vts) {
      tmp.resize(n * vt->d_out);
      std::memset(tmp.data(), 0, tmp.size() * sizeof(float));
      opq_alloc_time = std::chrono::high_resolution_clock::now() - opq_vt_start;
      vt->apply_noalloc(n, query_t.data(), tmp.data());
      opq_apply_time = std::chrono::high_resolution_clock::now() - opq_vt_start;
      query_t.resize(tmp.size());
      std::memcpy(query_t.data(), tmp.data(), tmp.size() * sizeof(float));
    }
  }
  auto opq_vt_end = std::chrono::high_resolution_clock::now();

  // step 1: find out the invlists for the query
  std::unique_ptr<faiss::idx_t[]> cq_ids =
      std::make_unique<faiss::idx_t[]>(params.nprobe * n);
  std::unique_ptr<float[]> cq_dist =
      std::make_unique<float[]>(params.nprobe * n);
  auto q_cq = (has_q_index_) ? q_cq_ : cq_;

  assert(q_cq->d == query_t.size() / n);
  q_cq->search(n, query_t.data(), params.nprobe, cq_dist.get(), cq_ids.get());
  auto cq_end = std::chrono::high_resolution_clock::now();

  // step 2: search the base index
  faiss::idx_t k_base = params.k_factor * params.k;
  std::unique_ptr<faiss::idx_t[]> base_lab(new faiss::idx_t[n * k_base]);
  std::unique_ptr<float[]> base_dist(new float[n * k_base]);
  faiss::IVFSearchParameters base_params;
  base_params.nprobe = params.nprobe;
  for (int i = 0; i < n; i++) {
    base_index_->search_preassigned_new(
        1, query_t.data() + i * d, k_base, cq_ids.get() + i * params.nprobe,
        cq_dist.get() + i * params.nprobe, base_dist.get() + i * k_base,
        base_lab.get() + i * k_base, false, &base_params, del_checker_.get());
  }

  // step 3: return the base search results
  distances->reset(base_dist.release());
  labels->reset(base_lab.release());

  auto base_end = std::chrono::high_resolution_clock::now();
  if (collect_stats_) {
    std::unique_lock lock(stats_mu_);
    stats_.accu_vt_time +=
        std::chrono::duration<double>(opq_vt_end - opq_vt_start).count();
    stats_.accu_pa_time +=
        std::chrono::duration<double>(cq_end - opq_vt_end).count();
    stats_.accu_qs_time +=
        std::chrono::duration<double>(base_end - cq_end).count();
    stats_.query_count++;
  }

  return true;
}

bool HakesIndex::SearchAndPA(int n, int d, const float* query,
                             const HakesSearchParams& params,
                             std::unique_ptr<float[]>* distances,
                             std::unique_ptr<faiss::idx_t[]>* labels,
                             std::unique_ptr<faiss::idx_t[]>* pa) {
  if (!distances || !labels || !pa) {
    throw std::runtime_error("distances, labels and pa must not be nullptr");
  }
  if (!keep_pa_) {
    throw std::runtime_error("PA is not kept in the index");
  }
  bool success = Search(n, d, query, params, distances, labels);
  if (!success) {
    return false;
  }
  auto k_base = params.k_factor * params.k;
  pa->reset(new faiss::idx_t[n * k_base]);
  std::unique_lock lock(mapping_mu_);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < k_base; j++) {
      auto it = pa_mapping_.find(labels->get()[i * k_base + j]);
      if (it != pa_mapping_.end()) {
        (*pa)[i * k_base + j] = it->second;
      } else {
        (*pa)[i * k_base + j] = -1;
      }
    }
  }
  return true;
}

// Each Rerank for a vector can have different sizes of base search found
// candidate points for this node.
bool HakesIndex::Rerank(int n, int d, const float* query, int k,
                        faiss::idx_t* k_base_count, faiss::idx_t* base_labels,
                        float* base_distances,
                        std::unique_ptr<float[]>* distances,
                        std::unique_ptr<faiss::idx_t[]>* labels) {
  if (!base_labels || !base_distances) {
    throw std::runtime_error(
        "base distances and base labels must not be nullptr");
  }

  if (!distances || !labels) {
    throw std::runtime_error("distances and labels must not be nullptr");
  }
  auto rerank_start = std::chrono::high_resolution_clock::now();
  faiss::idx_t total_labels = 0;
  std::vector<faiss::idx_t> base_label_start(n);
  for (int i = 0; i < n; i++) {
    base_label_start[i] = total_labels;
    total_labels += k_base_count[i];
  }

  // step 1(step 3 in V2): id translation
  {
    std::shared_lock lock(mapping_mu_);
    mapping_->get_val_for_ids(total_labels, base_labels, base_labels);
  }

  std::unique_ptr<float[]> dist(new float[n * k]);
  std::memset(dist.get(), 0, n * k * sizeof(float));
  std::unique_ptr<faiss::idx_t[]> lab(new faiss::idx_t[n * k]);
  std::memset(lab.get(), 0, n * k * sizeof(faiss::idx_t));

  // #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    // step 2 (step 4 in V2): search the refine index
    refine_index_->compute_distance_subset(1, query + i * d, k_base_count[i],
                                           base_distances + base_label_start[i],
                                           base_labels + base_label_start[i]);

    if (k >= k_base_count[i]) {
      std::memcpy(lab.get() + k * i, base_labels + base_label_start[i],
                  k_base_count[i] * sizeof(faiss::idx_t));
      std::memcpy(dist.get() + k * i, base_distances + base_label_start[i],
                  k_base_count[i] * sizeof(float));
      for (int j = k_base_count[i]; j < k; j++) {
        (lab.get() + k * i)[j] = -1;
      }
    }

    // step 3 (step 5 in V2): sort and store the result
    if (refine_index_->metric_type) {
      typedef faiss::CMax<float, faiss::idx_t> C;
      faiss::reorder_2_heaps<C>(1, (k > k_base_count[i]) ? k_base_count[i] : k,
                                lab.get() + k * i, dist.get() + k * i,
                                k_base_count[i],
                                base_labels + base_label_start[i],
                                base_distances + base_label_start[i]);

    } else if (refine_index_->metric_type == faiss::METRIC_INNER_PRODUCT) {
      typedef faiss::CMin<float, faiss::idx_t> C;
      faiss::reorder_2_heaps<C>(1, (k > k_base_count[i]) ? k_base_count[i] : k,
                                lab.get() + k * i, dist.get() + k * i,
                                k_base_count[i],
                                base_labels + base_label_start[i],
                                base_distances + base_label_start[i]);
    } else {
      FAISS_THROW_MSG("Metric type not supported");
    }
  }

  // step 4 (step 6 in V2): id translation
  {
    std::shared_lock lock(mapping_mu_);
    mapping_->get_keys_for_ids(n * k, lab.get(), lab.get());
  }

  // step 5 (step 7 in V2): return the result
  distances->reset(dist.release());
  labels->reset(lab.release());
  auto rerank_end = std::chrono::high_resolution_clock::now();
  if (collect_stats_) {
    std::lock_guard lock(stats_mu_);
    auto dur = std::chrono::duration<double>(rerank_end - rerank_start).count();
    stats_.accu_rerank_time += dur;
  }

  return true;
}

void HakesIndex::Reserve(faiss::idx_t n) {
  mapping_->reserve(n);
  refine_index_->reserve(n);
}

// same as EngineV2
bool HakesIndex::Checkpoint(const std::string& checkpoint_path) {
  std::filesystem::create_directories(checkpoint_path);
  std::filesystem::permissions(checkpoint_path,
                               std::filesystem::perms::owner_all |
                                   std::filesystem::perms::group_all |
                                   std::filesystem::perms::others_all,
                               std::filesystem::perm_options::add);

  if (base_index_) {
    std::string findex_path = checkpoint_path + "/" + FINDEX_NAME;
    std::unique_ptr<FileIOWriter> ff =
        std::make_unique<FileIOWriter>(findex_path.c_str());
    save_hakes_findex(ff.get(), this);
    if (has_q_index_) {
      std::string uindex_path = checkpoint_path + "/" + UINDEX_NAME;
      std::unique_ptr<FileIOWriter> uf =
          std::make_unique<FileIOWriter>(uindex_path.c_str());
      save_hakes_uindex(uf.get(), this);
    }

    if (keep_pa_) {
      std::string pa_mapping_path = checkpoint_path + "/" + PAMAP_NAME;
      std::unique_ptr<FileIOWriter> paf =
          std::make_unique<FileIOWriter>(pa_mapping_path.c_str());
      save_pa_map(paf.get(), this);
    }
  }
  if (refine_index_) {
    std::string rindex_path = checkpoint_path + "/" + RINDEX_NAME;
    std::unique_ptr<FileIOWriter> rf =
        std::make_unique<FileIOWriter>(rindex_path.c_str());
    save_hakes_rindex(rf.get(), this);
  }
  return true;
}

std::string HakesIndex::GetParams() const {
  faiss::StringIOWriter w;
  save_hakes_params(&w, this);
  return w.data;
}

bool HakesIndex::UpdateParams(const std::string& params) {
  faiss::StringIOReader r(params);
  std::unique_ptr<faiss::HakesIndex> loaded;
  load_hakes_params(&r, loaded.get());
  if (!loaded) {
    return false;
  }

  UpdateIndex(loaded.get());
  return true;
}

bool HakesIndex::DeleteWithIds(int n, const idx_t* ids) {
  if (del_checker_) {
    del_checker_->set(n, ids);
  }
  return true;
}

std::string HakesIndex::to_string() const {
  std::string ret = "HakesIndex:\nopq vt size " + std::to_string(vts_.size());
  for (auto vt : vts_) {
    ret += "\n  d_in " + std::to_string(vt->d_in) + ", d_out " +
           std::to_string(vt->d_out);
  }
  if (base_index_) {
    ret =
        ret + "\nbase_index n " + std::to_string(base_index_->ntotal) +
        ", nlist " + std::to_string(base_index_->nlist) + ", d " +
        std::to_string(base_index_->d) + ", pq m " +
        std::to_string(base_index_->pq.M) + ", nbits " +
        std::to_string(base_index_->pq.nbits) + ", metric: " +
        (base_index_->metric_type == faiss::METRIC_INNER_PRODUCT ? "ip" : "l2");
    ret = ret + "\nivf sq " + (use_ivf_sq_ ? "true" : "false");
    ret = ret + "\nearly termination " +
          (base_index_->use_early_termination_ ? "true" : "false") + " beta " +
          std::to_string(base_index_->et_params.beta) + " ce " +
          std::to_string(base_index_->et_params.ce);
    ret = ret + ((has_q_index_) ? "\nloaded q_index" : "\nno q_index");
  }
  if (refine_index_) {
    ret = ret + "\nmapping size " + std::to_string(mapping_->size()) +
          "\nrefine_index n " + std::to_string(refine_index_->ntotal) + ", d " +
          std::to_string(refine_index_->d) + ", metric: " +
          (refine_index_->metric_type == faiss::METRIC_INNER_PRODUCT ? "ip"
                                                                     : "l2");
  }
  // pa support port
  if (keep_pa_) {
    ret += "\npa mapping size " + std::to_string(pa_mapping_.size());
  }
  return ret;
};

}  // namespace faiss
