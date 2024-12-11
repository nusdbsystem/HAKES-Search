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

#include "fnr-worker/hakestask.h"

namespace hakes {

namespace {
void extract_search_task_response(const SearchResponse& batch_resp, int skip_n,
                                  HakesSearchTask* task) {
  task->resp_->status = batch_resp.status;
  task->resp_->msg =
      batch_resp.msg + "in batch " + std::to_string(batch_resp.n);
  auto n = task->req_viewer_.n;
  auto k_base = task->req_viewer_.k * task->req_viewer_.k_factor;
  task->resp_->n = n;
  task->resp_->k = k_base;

  task->resp_->scores_holder = std::unique_ptr<float[]>(new float[n * k_base]);
  memcpy(task->resp_->scores_holder.get(), batch_resp.scores + skip_n * k_base,
         sizeof(float) * n * k_base);
  task->resp_->scores = task->resp_->scores_holder.get();
  // task->resp_->scores = batch_resp.scores + skip_n * k_base;

  task->resp_->ids_holder = std::unique_ptr<int64_t[]>(new int64_t[n * k_base]);
  memcpy(task->resp_->ids_holder.get(), batch_resp.ids + skip_n * k_base,
         sizeof(int64_t) * n * k_base);
  task->resp_->ids = task->resp_->ids_holder.get();
  // task->resp_->ids = batch_resp.ids + skip_n * k_base;

  task->resp_->require_pa = task->req_viewer_.require_pa;
}

}  // anonymous namespace

bool HakesSearchTaskGroup::Run() {
  // // fast path for just one task: having a fast path seems to cause significant thoughput drop
  // if (size_ == 1) {
  //   HakesSearchTask* leader_task = static_cast<HakesSearchTask*>(leader_);
  //   executor_->Search(leader_task->req_viewer_, leader_task->resp_);
  //   return true;
  // }

  // printf("batch_search request: %s\n", batch_req_.to_string().c_str());
  Task* task_it = leader_;
  // build a new request
  batch_req_.vecs_holder =
      std::unique_ptr<float[]>(new float[batch_req_.n * batch_req_.d]);

  HakesSearchTask* leader_task = static_cast<HakesSearchTask*>(task_it);
  memcpy(
      batch_req_.vecs_holder.get(), leader_task->req_viewer_.vecs,
      sizeof(float) * leader_task->req_viewer_.n * leader_task->req_viewer_.d);
  int copied_n = leader_task->req_viewer_.n;

  while (task_it != last_task_) {
    HakesSearchTask* next_task = static_cast<HakesSearchTask*>(task_it->next_);

    memcpy(batch_req_.vecs_holder.get() + copied_n * batch_req_.d,
           next_task->req_viewer_.vecs,
           sizeof(float) * next_task->req_viewer_.n * next_task->req_viewer_.d);
    copied_n += next_task->req_viewer_.n;
    task_it = task_it->next_;
  }

  batch_req_.vecs = batch_req_.vecs_holder.get();
  executor_->Search(batch_req_, &batch_resp_);

  // assign the response to each task
  task_it = leader_;
  leader_task = static_cast<HakesSearchTask*>(task_it);
  extract_search_task_response(batch_resp_, 0, leader_task);
  int assigned_n = leader_task->req_viewer_.n;
  while (task_it != last_task_) {
    HakesSearchTask* next_task = static_cast<HakesSearchTask*>(task_it->next_);
    extract_search_task_response(batch_resp_, assigned_n, next_task);
    assigned_n += next_task->req_viewer_.n;
    task_it = task_it->next_;
  }

  return true;
}

}  // namespace hakes
