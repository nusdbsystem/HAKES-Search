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

#ifndef HAKES_SERVING_FNR_WORKER_HAKESENGINE_H_
#define HAKES_SERVING_FNR_WORKER_HAKESENGINE_H_

#include <atomic>
#include <mutex>

#include "faiss/ext/HakesIndex.h"
#include "fnr-worker/engine.h"
#include "fnr-worker/hakesexecutor.h"
#include "fnr-worker/hakestask.h"

namespace hakes {

class HakesBatchFnREngine : public FnREngine {
 public:
  HakesBatchFnREngine() : executor_(new HakesExecutor()) {};
  ~HakesBatchFnREngine() {}

  bool Initialize(const std::string& index_path, bool pa_mode = false, uint64_t cap = 0) override;

  bool IsInitialized() override;

  bool Add(const std::string& request, std::string* response) override;

  bool Search(const std::string& request, std::string* response) override;

  bool Rerank(const std::string& request, std::string* response) override;

  bool Checkpoint() override;

  bool GetIndex(std::string* response) override;

  bool UpdateIndex(const std::string& request, std::string* response) override;

  bool UpdateIndexLocal(const std::string& local_path) override;

  bool Close() {}

 private:
  std::string index_path_;
  std::atomic<uint32_t> index_version_;
  std::shared_mutex version_mutex_;
  std::unique_ptr<HakesExecutor> executor_;
  // task queues for each type of request: add for search option first
  std::unique_ptr<HakesSearchTaskQueue> search_task_queue_;
};

}  // namespace hakes

#endif  // HAKES_SERVING_FNR_WORKER_HAKESENGINE_H_