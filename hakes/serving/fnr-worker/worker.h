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

#ifndef HAKES_SERVING_FNR_WORKER_WORKER_H_
#define HAKES_SERVING_FNR_WORKER_WORKER_H_

#include <memory>

#include "fnr-worker/engine.h"
#include "server/serviceworker.h"

namespace hakes {
class FnRWorker : public ServiceWorker {
 public:
  FnRWorker(std::unique_ptr<FnREngine>&& engine) : engine_(std::move(engine)) {}

  ~FnRWorker();

  // delete copy and move constructors and assigment operators
  FnRWorker(const FnRWorker&) = delete;
  FnRWorker& operator=(const FnRWorker&) = delete;
  FnRWorker(FnRWorker&&) = delete;
  FnRWorker& operator=(FnRWorker&&) = delete;

  bool Initialize();

  bool Handle(const std::string& url, const std::string& input,
              std::string* output);

  void Close();

 private:
  std::unique_ptr<FnREngine> engine_;
};

}  // namespace hakes

#endif  // HAKES_SERVING_FNR_WORKER_WORKER_H_