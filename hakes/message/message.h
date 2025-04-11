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

#ifndef HAKES_SERVING_MESSAGE_H_
#define HAKES_SERVING_MESSAGE_H_

/**
 * @brief The request and resposne message format
 *
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#include "utils/hexutil.h"
#include "utils/json.h"

namespace hakes {

struct AddRequest {
  int n = 0;
  int d;
  float* vecs;
  std::unique_ptr<float[]> vecs_holder;
  int64_t* ids;
  std::unique_ptr<int64_t[]> ids_holder;

  std::string to_string() const {
    return "SearchRequest: n: " + std::to_string(n) +
           ", d: " + std::to_string(d);
  }
};

void encode_add_request(const AddRequest& request, std::string* data);

bool decode_add_request(const std::string& request_str, AddRequest* request);

struct AddResponse {
  bool status;
  std::string msg;
};

void encode_add_response(const AddResponse& response, std::string* data);

bool decode_add_response(const std::string& response_str,
                         AddResponse* response);

enum SearchMetricType : uint8_t {
  L2 = 0,
  IP = 1,
  COSINE = 2,
};

struct SearchRequest {
  int n;
  int d;
  float* vecs;
  std::unique_ptr<float[]> vecs_holder;
  int k;
  int nprobe;
  int k_factor;
  uint8_t metric_type;
  bool require_pa = false;

  std::string to_string() const {
    return "SearchRequest: n: " + std::to_string(n) +
           ", d: " + std::to_string(d) + ", k: " + std::to_string(k) +
           ", nprobe: " + std::to_string(nprobe) +
           ", k_factor: " + std::to_string(k_factor) +
           ", metric_type: " + std::to_string(metric_type) +
           ", require_pa: " + std::to_string(require_pa);
  }
};

void encode_search_request(const SearchRequest& request, std::string* data);

bool decode_search_request(const std::string& request_str,
                           SearchRequest* request);

struct SearchResponse {
  bool status;
  std::string msg;
  int n;
  int k;
  float* scores;
  std::unique_ptr<float[]> scores_holder;
  int64_t* ids;
  std::unique_ptr<int64_t[]> ids_holder;
  bool require_pa = false;
  int64_t* pas;
  std::unique_ptr<int64_t[]> pas_holder;
  int index_version = -1;
};

void encode_search_response(const SearchResponse& response, std::string* data);

bool decode_search_response(const std::string& response_str,
                            SearchResponse* response);

struct DeleteRequest {
  int n = 0;
  int64_t* ids;
  std::unique_ptr<int64_t[]> ids_holder;
};

void encode_delete_request(const DeleteRequest& request, std::string* data);

bool decode_delete_request(const std::string& request_str,
                           DeleteRequest* request);

struct DeleteResponse {
  bool status;
  std::string msg;
};

void encode_delete_response(const DeleteResponse& response, std::string* data);

bool decode_delete_response(const std::string& response_str,
                            DeleteResponse* response);

}  // namespace hakes

#endif  // HAKES_SERVING_MESSAGE_H_
