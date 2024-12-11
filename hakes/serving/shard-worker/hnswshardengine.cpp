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

#include "shard-worker/hnswshardengine.h"

#include <filesystem>

#include "common/checkpoint.h"
#include "message/message.h"
#include "message/message_ext.h"

namespace {
const char* hnsw_index_name = "hnsw.bin";
}

namespace hakes {
bool HnswShardEngine::Initialize(const std::string& index_path) {
  auto checkpoint = get_latest_checkpoint_path(index_path);
  if (checkpoint.empty()) {
    int m = 32;
    int efc = 200;
    printf("no checkpoint found, initialize new hnsw index");
    hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(), capacity_, m, efc);
    printf("hnsw index initialized with m %d and efc %d\n", m, efc);
    index_version_ = 0;
  } else {
    std::string hnsw_index_path =
        index_path + "/" + checkpoint + "/" + hnsw_index_name;
    if (!std::filesystem::exists(hnsw_index_path)) {
      throw std::runtime_error("No hnsw index found in " + hnsw_index_path);
    }
    hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(), hnsw_index_path.c_str(), false, capacity_);
    index_version_ = get_checkpoint_no(checkpoint);
    printf("hnsw index loaded with m %d and efc %d\n", hnsw_->M_,
           hnsw_->ef_construction_);
  }
  hnsw_->setEf(efs_);

  initialized_.store(true, std::memory_order_relaxed);
  index_path_ = index_path;
  return true;
}

bool HnswShardEngine::IsInitialized() {
  return initialized_.load(std::memory_order_relaxed);
}

bool HnswShardEngine::Add(const std::string& request, std::string* response) {
  auto start_time = std::chrono::high_resolution_clock::now();
  assert(response);
  AddResponse add_response;

  // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  // upon error return 502 (500 can be used for framework internal error)

  // decode the request
  AddRequest add_request;
  bool success = decode_add_request(request, &add_request);
  if (!success) {
    add_response.status = false;
    add_response.msg = "decode add request error";
    encode_add_response(add_response, response);
    return false;
  }
  // decode the request

  int n = add_request.n;
  int d = add_request.d;
  float* vecs = add_request.vecs;
  int64_t* ids = add_request.ids;

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    hnsw_->addPoint(vecs + i * d, ids[i]);
  }
  add_response.status = true;
  add_response.msg = "add success";
  encode_add_response(add_response, response);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  printf("Add time: %ld us\n", duration.count());

  // return 200 with the response
  return true;
}

bool HnswShardEngine::Search(const std::string& request,
                             std::string* response) {
  auto start_time = std::chrono::high_resolution_clock::now();
  assert(response);
  SearchResponse search_response;
  // printf("Request (len=%ld): %s", request.size(), request.c_str());

  // std::this_thread::sleep_for(std::chrono::milliseconds(10));
  // upon error return 502 (500 can be used for framework internal error)

  // decode the request
  SearchRequest search_request;
  bool success = decode_search_request(request, &search_request);
  if (!success) {
    search_response.status = false;
    search_response.msg = "decode search request error";
    encode_search_response(search_response, response);
    return false;
  }

  switch (search_request.metric_type) {
    case SearchMetricType::COSINE:
    case SearchMetricType::IP:
      break;
    default:
      search_response.status = false;
      search_response.msg = "unsupported metric type";
      encode_search_response(search_response, response);
      return false;
  }

  int n = search_request.n;
  int d = search_request.d;
  float* query = search_request.vecs;
  int k = search_request.k;

  std::unique_ptr<float[]> distances =
      std::unique_ptr<float[]>(new float[n * k]);
  std::unique_ptr<int64_t[]> labels =
      std::unique_ptr<int64_t[]>(new int64_t[n * k]);

#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    auto result = hnsw_->searchKnn(query + i * d, k);
    auto start_idx = i * k;
    while (result.size() > 0) {
      auto top = result.top();
      // hnsw return 1-ip. convert to ip
      distances.get()[start_idx] = 1.0f - top.first;
      labels.get()[start_idx] = top.second;
      start_idx++;
      result.pop();
    }
    while (start_idx < (i + 1) * k) {
      distances.get()[start_idx] = -1;
      labels.get()[start_idx] = -1;
      start_idx++;
    }
  }

  // wrap the result into a response json
  search_response.status = true;
  search_response.msg = "search success";
  search_response.n = search_request.n;
  search_response.k = search_request.k;
  search_response.scores_holder.reset(distances.release());
  search_response.scores = search_response.scores_holder.get();
  search_response.ids_holder.reset(labels.release());
  search_response.ids = search_response.ids_holder.get();

  encode_search_response(search_response, response);

  // printf("Response (len=%ld): %s", response->size(), response->c_str());
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  printf("Search time: %ld us\n", duration.count());
  // return 200 with the response
  return true;
}

bool HnswShardEngine::Checkpoint() {
  if (!initialized_.load(std::memory_order_relaxed)) {
    return false;
  }
  auto checkpoint_path =
      index_path_ + "/" +
      format_checkpoint_path(index_version_.fetch_add(1) + 1);

  std::filesystem::create_directories(checkpoint_path.c_str());
  std::filesystem::permissions(checkpoint_path.c_str(),
                               std::filesystem::perms::owner_all |
                                   std::filesystem::perms::group_all |
                                   std::filesystem::perms::others_all,
                               std::filesystem::perm_options::add);

  hnsw_->saveIndex(checkpoint_path + "/" + hnsw_index_name);
  return true;
}

bool HnswShardEngine::GetIndex(std::string* response) {
  throw std::runtime_error("Not implemented");
}

bool HnswShardEngine::UpdateIndex(const std::string& request,
                                  std::string* response) {
  throw std::runtime_error("Not implemented");
}

}  // namespace hakes
