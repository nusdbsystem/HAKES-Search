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

#ifndef HAKES_SERVING_SHARD_WORKER_WORKER_H_
#define HAKES_SERVING_SHARD_WORKER_WORKER_H_

#include <memory>

#include "server/serviceworker.h"
#include "shard-worker/engine.h"

namespace hakes {
class ShardWorker : public ServiceWorker {
 public:
  ShardWorker(std::unique_ptr<ShardEngine>&& engine)
      : engine_(std::move(engine)) {}

  ~ShardWorker();

  // delete copy and move constructors and assigment operators
  ShardWorker(const ShardWorker&) = delete;
  ShardWorker& operator=(const ShardWorker&) = delete;
  ShardWorker(ShardWorker&&) = delete;
  ShardWorker& operator=(ShardWorker&&) = delete;

  bool Initialize();

  bool Handle(const std::string& url, const std::string& input,
              std::string* output);

  void Close();

 private:
  std::unique_ptr<ShardEngine> engine_;
};

}  // namespace hakes

#endif  // HAKES_SERVING_SHARD_WORKER_WORKER_H_