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

#ifndef HAKES_SERVING_SHARD_WORKER_HNSWSHARDENGINE_H_
#define HAKES_SERVING_SHARD_WORKER_HNSWSHARDENGINE_H_

#include <atomic>
#include <mutex>
#include <shared_mutex>

#include "hnswlib/hnswlib.h"
#include "shard-worker/engine.h"

namespace hakes {

class HnswShardEngine : public ShardEngine {
 public:
  HnswShardEngine(int d, uint64_t cap, int efs = 512)
      : d_(d),
        capacity_(cap),
        efs_(efs),
        space_(new hnswlib::InnerProductSpace(d_)) {};
  ~HnswShardEngine() {}

  bool Initialize(const std::string& index_path) override;

  bool IsInitialized() override;

  bool Add(const std::string& request, std::string* response) override;

  bool Search(const std::string& request, std::string* response) override;

  bool Checkpoint() override;

  bool GetIndex(std::string* response) override;

  bool UpdateIndex(const std::string& request, std::string* response) override;

  bool Close() {}

 private:
  // std::unique_ptr<faiss::IndexFlatL> index_;
  std::atomic_bool initialized_{false};
  std::shared_mutex mutex_;
  std::string index_path_;
  std::atomic<uint32_t> index_version_;

  int d_;
  uint64_t capacity_;
  int efs_;

  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
};
}  // namespace hakes
#endif  // HAKES_SERVING_SHARD_WORKER_HNSWSHARDENGINE_H_