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

#include "fnr-worker/hakesengine.h"

#include "common/checkpoint.h"
#include "message/message.h"
#include "message/message_ext.h"

namespace hakes {

bool HakesFnREngine::Initialize(const std::string& index_path, bool pa_mode, uint64_t cap) {
  auto checkpoint = get_latest_checkpoint_path(index_path);
  if (checkpoint.empty()) {
    throw std::runtime_error("No checkpoint found in " + index_path);
  }

  bool success = executor_->Initialize(index_path + "/" + checkpoint, pa_mode, cap);
  if (!success) {
    return false;
  }

  index_path_ = index_path;
  index_version_ = get_checkpoint_no(checkpoint);
  return true;
}

bool HakesFnREngine::IsInitialized() { return executor_->IsInitialized(); }

bool HakesFnREngine::Add(const std::string& request, std::string* response) {
  auto start_time = std::chrono::high_resolution_clock::now();
  assert(response);
  ExtendedAddResponse add_response;

  // decode the request
  ExtendedAddRequest add_request;
  bool success = decode_extended_add_request(request, &add_request);
  if (!success) {
    add_response.status = false;
    add_response.msg = "decode add request error";
    encode_add_response(add_response, response);
    return false;
  }
  // decode the request

  // execute the logic
  auto add_core_start = std::chrono::high_resolution_clock::now();
  success = executor_->Add(add_request, &add_response);
  auto add_core_end = std::chrono::high_resolution_clock::now();

  if (!success) {
    encode_add_response(add_response, response);
    return false;
  }
  encode_extended_add_response(add_response, response);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  auto core_duration = std::chrono::duration_cast<std::chrono::microseconds>(
      add_core_end - add_core_start);
  printf("Add time: %ld us\n", duration.count());

  // return 200 with the response
  return true;
}

bool HakesFnREngine::Search(const std::string& request, std::string* response) {
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

  success = executor_->Search(search_request, &search_response);

  encode_search_response(search_response, response);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  // return 200 with the response
  return success;
}

bool HakesFnREngine::Rerank(const std::string& request, std::string* response) {
  auto start_time = std::chrono::high_resolution_clock::now();
  assert(response);
  SearchResponse search_response;

  // decode the request
  RerankRequest rerank_request;
  bool success = decode_rerank_request(request, &rerank_request);
  if (!success) {
    search_response.status = false;
    search_response.msg = "decode rerank request error";
    encode_search_response(search_response, response);
    return false;
  }

  success = executor_->Rerank(rerank_request, &search_response);
  encode_search_response(search_response, response);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  // return 200 with the response
  return success;
}

bool HakesFnREngine::Checkpoint() {
  std::shared_lock lock(version_mutex_);
  std::string checkpoint_path =
      index_path_ + "/" +
      format_checkpoint_path(
          index_version_.fetch_add(1, std::memory_order_relaxed) + 1);
  return executor_->Checkpoint(std::move(checkpoint_path));
}

bool HakesFnREngine::GetIndex(std::string* response) {
  GetIndexResponse get_index_response;
  {
    std::shared_lock lock(version_mutex_);
    uint32_t index_version = index_version_.load(std::memory_order_relaxed);
    if (executor_->GetIndex(&get_index_response)) {
      get_index_response.index_version = index_version;
    }
  }
  encode_get_index_response(get_index_response, response);
  return true;
}

bool HakesFnREngine::UpdateIndex(const std::string& request,
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
    std::unique_lock lock(version_mutex_);
    bool success =
        executor_->UpdateIndex(update_index_request, &update_index_response);
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

bool HakesFnREngine::UpdateIndexLocal(const std::string& local_path) {
  printf("UpdateIndexLocal: %s\n", local_path.c_str());
  return executor_->UpdateIndexLocal(local_path);
}

}  // namespace hakes
