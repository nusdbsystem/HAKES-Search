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

#ifndef HAKES_SERVING_FNRWORKER_HAKESTASK_H_
#define HAKES_SERVING_FNRWORKER_HAKESTASK_H_

#include "faiss/ext/HakesIndex.h"
#include "fnr-worker/hakesexecutor.h"
#include "message/message.h"
#include "task-queue/task.h"

namespace hakes {

class HakesSearchTask : public Task {
 public:
  HakesSearchTask(const SearchRequest& request, SearchResponse* resp)
      : Task(), req_viewer_(request), resp_(resp) {}
  ~HakesSearchTask() {}

  std::string to_string() override {
    return "HakesSearchTask: " + req_viewer_.to_string();
  }

  const SearchRequest& req_viewer_;
  SearchResponse* resp_;
};

class HakesSearchTaskGroup : public TaskGroup {
 public:
  HakesSearchTaskGroup(HakesExecutor* executor)
      : TaskGroup(), executor_(executor) {}
  ~HakesSearchTaskGroup() {}

  bool Run() override;

  inline void SetLeader(Task* leader) override {
    this->TaskGroup::SetLeader(leader);
    HakesSearchTask* leader_task = static_cast<HakesSearchTask*>(leader);
    batch_req_.n = 0;
    batch_req_.d = leader_task->req_viewer_.d;
    batch_req_.k = leader_task->req_viewer_.k;
    batch_req_.nprobe = leader_task->req_viewer_.nprobe;
    batch_req_.k_factor = leader_task->req_viewer_.k_factor;
    batch_req_.require_pa = leader_task->req_viewer_.require_pa;
    batch_req_.metric_type = leader_task->req_viewer_.metric_type;
  }

  inline bool ClaimTask(Task* task) override {
    HakesSearchTask* hakes_task = static_cast<HakesSearchTask*>(task);
    if (hakes_task->req_viewer_.require_pa != batch_req_.require_pa) {
      return false;
    }
    batch_req_.n += hakes_task->req_viewer_.n;
    // now we issue homogeneous requests
    assert(hakes_task->req_viewer_.d == batch_req_.d);
    assert(hakes_task->req_viewer_.k == batch_req_.k);
    assert(hakes_task->req_viewer_.nprobe == batch_req_.nprobe);
    assert(hakes_task->req_viewer_.k_factor == batch_req_.k_factor);
    assert(hakes_task->req_viewer_.metric_type == batch_req_.metric_type);
    return this->TaskGroup::ClaimTask(task);
  }

  SearchRequest batch_req_;
  SearchResponse batch_resp_;
  HakesExecutor* executor_;
};

class HakesSearchTaskQueue : public TaskQueue {
 public:
  HakesSearchTaskQueue(HakesExecutor* executor)
      : TaskQueue(), executor_(executor) {}
  ~HakesSearchTaskQueue() {}

  std::unique_ptr<TaskGroup> create_group() override {
    return std::make_unique<HakesSearchTaskGroup>(executor_);
  }

  HakesExecutor* executor_;
};

}  // namespace hakes

#endif  // HAKES_SERVING_FNRWORKER_HAKESTASK_H_
