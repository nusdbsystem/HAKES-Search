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

#include "fnr-worker/hakesexecutor.h"

namespace hakes {

bool HakesExecutor::Initialize(const std::string& index_path, int mode,
                               bool pa_mode, uint64_t cap) {
  index_ = std::make_unique<faiss::HakesIndex>();
  index_->use_ivf_sq_ = true;
  if (!index_->Initialize(index_path, mode, pa_mode)) {
    return false;
  }
  if (cap > 0) {
    index_->Reserve(cap);
  }
  return true;
}

bool HakesExecutor::IsInitialized() { return index_ != nullptr; }

bool HakesExecutor::Add(const ExtendedAddRequest& request,
                        ExtendedAddResponse* response) {
  std::unique_ptr<faiss::idx_t[]> assign;
  int vecs_t_d = 0;
  std::unique_ptr<float[]> transformed_vecs;
  bool success = false;
  if (request.add_to_refine_only) {
    success = index_->AddToRefine(request.n, request.d, request.vecs,
                                  static_cast<faiss::idx_t*>(request.ids),
                                  static_cast<faiss::idx_t*>(request.assign));
  } else if (request.assigned) {
    std::shared_lock lock(params_mutex_);
    success = index_->AddBasePreassigned(
        request.n, request.d, request.vecs,
        static_cast<faiss::idx_t*>(request.ids), request.assign);
  } else {
    std::shared_lock lock(params_mutex_);
    assign = std::make_unique<faiss::idx_t[]>(request.n);
    success = index_->AddWithIds(request.n, request.d, request.vecs,
                                 static_cast<faiss::idx_t*>(request.ids),
                                 assign.get(), &vecs_t_d, &transformed_vecs);

    printf("after addition: %d", index_->base_index_.get()->ntotal);
  }

  if (!success) {
    response->status = false;
    response->msg = "add error";
    return false;
  }
  response->status = true;
  response->msg = "add success";
  response->n = request.n;
  response->assign_holder = std::move(assign);
  response->assign = response->assign_holder.get();
  if (vecs_t_d > 0) {
    response->vecs_t_d = vecs_t_d;
    response->vecs_t_holder = std::move(transformed_vecs);
    response->vecs_t = response->vecs_t_holder.get();
  } else {
    response->vecs_t_d = 0;
    response->vecs_t = nullptr;
  }
  return true;
}

bool HakesExecutor::Search(const SearchRequest& request,
                           SearchResponse* response) {
  faiss::MetricType mt;
  switch (request.metric_type) {
    case SearchMetricType::L2:
      mt = faiss::METRIC_L2;
      break;
    case SearchMetricType::COSINE:
    case SearchMetricType::IP:
      mt = faiss::METRIC_INNER_PRODUCT;
      break;
    default:
      response->status = false;
      response->msg = "unsupported metric type";
      return false;
  }

  auto search_params = faiss::HakesSearchParams{
      .nprobe = request.nprobe,
      .k = request.k,
      .k_factor = request.k_factor,
      .metric_type = mt,
  };
  // decode the request

  // execute the logic
  std::unique_ptr<float[]> distances;
  std::unique_ptr<faiss::idx_t[]> labels;
  bool ret;

  {
    std::shared_lock lock(params_mutex_);
    if (request.require_pa) {
      std::unique_ptr<faiss::idx_t[]> pas;
      ret = index_->SearchAndPA(request.n, request.d, request.vecs,
                                search_params, &distances, &labels, &pas);
      if (ret) {
        response->require_pa = true;
        response->pas_holder = std::move(pas);
        response->pas = response->pas_holder.get();
      }
    } else {
      ret = index_->Search(request.n, request.d, request.vecs, search_params,
                           &distances, &labels);
    }
  }

  if (!ret) {
    response->status = false;
    response->msg = "search error";
    return false;
  }

  // wrap the result into a response json
  response->status = true;
  response->msg = "search success";
  response->n = request.n;
  // the difference with EngineV1 and EngineV2 as we are returning base search
  // results.
  response->k = request.k * request.k_factor;
  response->scores_holder.reset(distances.release());
  response->scores = response->scores_holder.get();
  response->ids_holder.reset(labels.release());
  response->ids = response->ids_holder.get();
  return true;
}

bool HakesExecutor::Rerank(const RerankRequest& request,
                           SearchResponse* response) {
  faiss::MetricType mt;
  switch (request.metric_type) {
    case SearchMetricType::L2:
      mt = faiss::METRIC_L2;
      break;
    case SearchMetricType::COSINE:
    case SearchMetricType::IP:
      mt = faiss::METRIC_INNER_PRODUCT;
      break;
    default:
      response->status = false;
      response->msg = "unsupported metric type";
      return false;
  }
  // decode the request

  // execute the logic
  std::unique_ptr<float[]> distances;
  std::unique_ptr<faiss::idx_t[]> labels;
  auto ret = index_->Rerank(request.n, request.d, request.vecs, request.k,
                            request.k_base_count, request.base_labels,
                            request.base_distances, &distances, &labels);
  if (!ret) {
    response->status = false;
    response->msg = "rerank error";
    return false;
  }

  // wrap the result into a response json
  response->status = true;
  response->msg = "rerank success";
  response->n = request.n;
  response->k = request.k;
  response->scores_holder.reset(distances.release());
  response->scores = response->scores_holder.get();
  response->ids_holder.reset(labels.release());
  response->ids = response->ids_holder.get();
  return true;
}

bool HakesExecutor::Delete(const DeleteRequest& request,
                           DeleteResponse* response) {
  {
    std::shared_lock lock(params_mutex_);
    if (!index_->DeletionEnabled()) {
      lock.unlock();
      std::unique_lock wl(params_mutex_);
      if (!index_->DeletionEnabled()) {
        index_->EnableDeletion();
      }
    }
  }
  index_->DeleteWithIds(request.n, request.ids);
  response->status = true;
  response->msg = "delete success";
  return true;
}

bool HakesExecutor::Checkpoint(const std::string& checkpoint_path) {
  std::shared_lock lock(params_mutex_);
  return index_->Checkpoint(std::move(checkpoint_path));
}

bool HakesExecutor::GetIndex(GetIndexResponse* response) {
  std::string params;
  {
    std::shared_lock lock(params_mutex_);
    params.assign(index_->GetParams());
  }

  if (params.empty()) {
    response->status = false;
    response->msg = "get index error";
    return false;
  } else {
    response->status = true;
    response->msg = "get index success";
    response->params = std::move(params);
    return true;
  }
}

bool HakesExecutor::UpdateIndex(const UpdateIndexRequest& request,
                                UpdateIndexResponse* /*response*/) {
  std::unique_lock lock(params_mutex_);
  return index_->UpdateParams(request.params);
}

bool HakesExecutor::UpdateIndexLocal(const std::string& local_path) {
  std::unique_lock lock(params_mutex_);
  faiss::HakesIndex loaded;
  loaded.use_ivf_sq_ = true;
  loaded.Initialize(local_path);
  index_->UpdateIndex(&loaded);
  return true;
}

}  // namespace hakes