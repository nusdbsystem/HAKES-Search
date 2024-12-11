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

#ifndef HAKES_SERVING_FNRWORKER_HAKESEXECUTOR_H_
#define HAKES_SERVING_FNRWORKER_HAKESEXECUTOR_H_

#include <memory>

#include "faiss/ext/HakesIndex.h"
#include "message/message.h"
#include "message/message_ext.h"

namespace hakes {

class HakesExecutor {
 public:
  HakesExecutor() = default;
  ~HakesExecutor() {}

  bool Initialize(const std::string& index_path, bool pa_mode = false,
                  uint64_t cap = 0);

  bool IsInitialized();

  bool Add(const ExtendedAddRequest& request, ExtendedAddResponse* response);

  bool Search(const SearchRequest& request, SearchResponse* response);

  bool Rerank(const RerankRequest& request, SearchResponse* response);

  bool Checkpoint(const std::string& checkpoint_path);

  bool GetIndex(GetIndexResponse* response);

  bool UpdateIndex(const UpdateIndexRequest& request,
                   UpdateIndexResponse* response);

  bool UpdateIndexLocal(const std::string& local_path);

  bool Close() {}

 private:
  std::shared_mutex params_mutex_;
  std::unique_ptr<faiss::HakesIndex> index_;
};
}  // namespace hakes

#endif  // HAKES_SERVING_FNRWORKER_HAKESEXECUTOR_H_
