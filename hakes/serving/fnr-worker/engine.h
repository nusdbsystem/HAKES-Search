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

#ifndef HAKES_SERVING_FNR_WORKER_ENGINE_H_
#define HAKES_SERVING_FNR_WORKER_ENGINE_H_

#include <cstddef>
#include <string>

namespace hakes {

class FnREngine {
 public:
  FnREngine() = default;

  virtual ~FnREngine() = default;

  // delete copy constructor and assignment operator
  FnREngine(const FnREngine&) = delete;
  FnREngine& operator=(const FnREngine&) = delete;
  // delete move constructor and assignment operator
  FnREngine(FnREngine&&) = delete;
  FnREngine& operator=(FnREngine&&) = delete;

  virtual bool Initialize(const std::string& index_path, int mode, bool pa_mode,
                          uint64_t cap = 0) = 0;

  virtual bool IsInitialized() = 0;

  virtual bool Add(const std::string& request, std::string* response) = 0;

  virtual bool Search(const std::string& request, std::string* response) = 0;

  virtual bool Rerank(const std::string& request, std::string* response) = 0;

  virtual bool Delete(const std::string& request, std::string* response) = 0;

  virtual bool Checkpoint() = 0;

  virtual bool GetIndex(std::string* response) = 0;

  virtual bool UpdateIndex(const std::string& request,
                           std::string* response) = 0;

  virtual bool UpdateIndexLocal(const std::string& local_path) = 0;

  virtual bool Close() = 0;
};

}  // namespace hakes
#endif  // HAKES_SERVING_FNR_WORKER_ENGINE_H_
