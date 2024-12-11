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

#include "shard-worker/hakesshardengine.h"

#include <vector>

#include "common/checkpoint.h"
#include "message/message.h"
#include "message/message_ext.h"

namespace hakes {

bool HakesShardEngine::Initialize(const std::string& index_path) {
  auto checkpoint = get_latest_checkpoint_path(index_path);
  if (checkpoint.empty()) {
    throw std::runtime_error("No checkpoint found in " + index_path);
  }

  index_ = std::make_unique<faiss::HakesIndex>();
  if (!index_->Initialize(index_path + "/" + checkpoint)) {
    return false;
  }
  index_path_ = index_path;
  index_version_ = get_checkpoint_no(checkpoint);
  return true;
}

bool HakesShardEngine::IsInitialized() { return index_ != nullptr; }

bool HakesShardEngine::Add(const std::string& request, std::string* response) {
  auto start_time = std::chrono::high_resolution_clock::now();
  assert(response);
  AddResponse add_response;

  // decode the request
  AddRequest add_request;
  // bool success = decode_extended_add_request(request, &add_request);
  bool success = decode_add_request(request, &add_request);
  if (!success) {
    add_response.status = false;
    add_response.msg = "decode add request error";
    encode_add_response(add_response, response);
    return false;
  }
  // decode the request

  // execute the logic
  std::unique_ptr<faiss::idx_t[]> assign;
  int vecs_t_d = 0;
  std::unique_ptr<float[]> transformed_vecs;
  auto add_core_start = std::chrono::high_resolution_clock::now();
  // if (add_request.add_to_refine_only) {
  //   success =
  //       index_->AddToRefine(add_request.n, add_request.d, add_request.vecs,
  //                           static_cast<faiss::idx_t*>(add_request.ids),
  //                           static_cast<faiss::idx_t*>(add_request.assign));
  // } else if (add_request.assigned) {
  //   std::shared_lock lock(mutex_);
  //   success = index_->AddBasePreassigned(
  //       add_request.n, add_request.d, add_request.vecs,
  //       static_cast<faiss::idx_t*>(add_request.ids), add_request.assign);
  // } else {
  std::shared_lock lock(mutex_);
  assign = std::make_unique<faiss::idx_t[]>(add_request.n);
  success = index_->AddWithIds(add_request.n, add_request.d, add_request.vecs,
                               static_cast<faiss::idx_t*>(add_request.ids),
                               assign.get(), &vecs_t_d, &transformed_vecs);
  // }
  auto add_core_end = std::chrono::high_resolution_clock::now();

  if (!success) {
    add_response.status = false;
    add_response.msg = "add error";
    encode_add_response(add_response, response);
    return false;
  }
  add_response.status = true;
  add_response.msg = "add success";
  // add_response.n = add_request.n;
  // add_response.assign_holder = std::move(assign);
  // add_response.assign = add_response.assign_holder.get();
  // if (vecs_t_d > 0) {
  //   add_response.vecs_t_d = vecs_t_d;
  //   add_response.vecs_t_holder = std::move(transformed_vecs);
  //   add_response.vecs_t = add_response.vecs_t_holder.get();
  // } else {
  //   add_response.vecs_t_d = 0;
  //   add_response.vecs_t = nullptr;
  // }
  // encode_extended_add_response(add_response, response);
  encode_add_response(add_response, response);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  auto core_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      add_core_end - add_core_start);
  printf("Add time: %ld us\n", duration.count());

  // return 200 with the response
  return true;
}

bool HakesShardEngine::Search(const std::string& request,
                              std::string* response) {
  auto start_time = std::chrono::high_resolution_clock::now();
  assert(response);
  SearchResponse search_response;

  // decode the request
  SearchRequest search_request;
  bool success = decode_search_request(request, &search_request);
  if (!success) {
    search_response.status = false;
    search_response.msg = "decode search request error";
    encode_search_response(search_response, response);
    return false;
  }

  faiss::MetricType mt;
  switch (search_request.metric_type) {
    case SearchMetricType::L2:
      mt = faiss::METRIC_L2;
      break;
    case SearchMetricType::COSINE:
    case SearchMetricType::IP:
      mt = faiss::METRIC_INNER_PRODUCT;
      break;
    default:
      search_response.status = false;
      search_response.msg = "unsupported metric type";
      encode_search_response(search_response, response);
      return false;
  }

  auto search_params = faiss::HakesSearchParams{
      .nprobe = search_request.nprobe,
      .k = search_request.k,
      .k_factor = search_request.k_factor,
      .metric_type = mt,
  };
  // decode the request

  // execute the logic
  std::unique_ptr<float[]> distances;
  std::unique_ptr<faiss::idx_t[]> labels;
  bool ret;

  {
    std::shared_lock lock(mutex_);
    if (search_request.require_pa) {
      std::unique_ptr<faiss::idx_t[]> pas;
      ret = index_->SearchAndPA(search_request.n, search_request.d,
                                search_request.vecs, search_params, &distances,
                                &labels, &pas);
      if (ret) {
        search_response.require_pa = true;
        search_response.pas_holder = std::move(pas);
        search_response.pas = search_response.pas_holder.get();
      }
    } else {
      ret = index_->Search(search_request.n, search_request.d,
                           search_request.vecs, search_params, &distances,
                           &labels);
    }
  }

  if (!ret) {
    search_response.status = false;
    search_response.msg = "search error";
    encode_search_response(search_response, response);
    return false;
  }

  // refine followed by search for shard engine
  std::vector<int64_t> k_base_count(search_request.n,
                                    search_request.k * search_request.k_factor);
  ret = index_->Rerank(search_request.n, search_request.d, search_request.vecs,
                       search_request.k, k_base_count.data(), labels.get(),
                       distances.get(), &distances, &labels);

  if (!ret) {
    search_response.status = false;
    search_response.msg = "rerank error";
    encode_search_response(search_response, response);
    return false;
  }

  // wrap the result into a response json
  search_response.status = true;
  search_response.msg = "search success";
  search_response.n = search_request.n;
  // the difference with EngineV1 and EngineV2 as we are returning base search
  // results.
  search_response.k = search_request.k;
  search_response.scores_holder.reset(distances.release());
  search_response.scores = search_response.scores_holder.get();
  search_response.ids_holder.reset(labels.release());
  search_response.ids = search_response.ids_holder.get();

  encode_search_response(search_response, response);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  printf("Search time: %ld us\n", duration.count());

  // return 200 with the response
  return true;
}

bool HakesShardEngine::Checkpoint() {
  std::string checkpoint_path =
      index_path_ + "/" +
      format_checkpoint_path(
          index_version_.fetch_add(1, std::memory_order_relaxed) + 1);
  std::shared_lock lock(mutex_);
  return index_->Checkpoint(checkpoint_path);
}

bool HakesShardEngine::GetIndex(std::string* response) {
  uint32_t index_version;
  std::string params;
  {
    std::shared_lock lock(mutex_);
    index_version = index_version_.load(std::memory_order_relaxed);
    params.assign(index_->GetParams());
  }

  GetIndexResponse get_index_response;
  get_index_response.status = !params.empty();
  get_index_response.index_version = index_version;
  get_index_response.params = params;

  encode_get_index_response(get_index_response, response);
  return true;
}

bool HakesShardEngine::UpdateIndex(const std::string& request,
                                   std::string* response) {
  auto start_time = std::chrono::high_resolution_clock::now();
  assert(response);
  UpdateIndexResponse update_index_response;

  // decode the request
  UpdateIndexRequest update_index_request;
  bool success = decode_update_index_request(request, &update_index_request);
  if (!success) {
    update_index_response.status = false;
    update_index_response.msg = "decode rerank request error";
    encode_update_index_response(update_index_response, response);
    return false;
  }
  {
    std::unique_lock lock(mutex_);
    bool success = index_->UpdateParams(update_index_request.params);
    // only checkpoint bump index version
    update_index_response.index_version =
        (success) ? index_version_.load(std::memory_order_relaxed) : -1;
  }

  if (update_index_response.index_version == -1) {
    update_index_response.status = false;
    update_index_response.msg = "update index error";
    encode_update_index_response(update_index_response, response);
    return false;
  } else {
    update_index_response.status = true;
    update_index_response.msg = "update index success";
    encode_update_index_response(update_index_response, response);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    printf("UpdateIndex time: %ld us\n", duration.count());
    return true;
  }
}

}  // namespace hakes