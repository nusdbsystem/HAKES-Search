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

#include "task.h"

#include <cassert>
#include <thread>

namespace hakes {
namespace {

// only called by a leader
void BacktrackCreateNextLink(Task* task) {
  Task* prev = task->prev_;
  while (prev && prev->next_ == nullptr) {
    prev->next_ = task;
    task = task->prev_;
    prev = task->prev_;
  }
}

}  // anonymous namespace

void TaskQueue::Schedule(Task* task) {
  assert(task);

  Task* newest_task = newest_task_.load(std::memory_order_relaxed);
  while (true) {
    task->prev_ = newest_task;
    if (newest_task_.compare_exchange_weak(newest_task, task)) {
      if (newest_task == nullptr) {
        task->state_.store(Task::kLeader, std::memory_order_relaxed);
      }
      break;
    }
  }

  if (task->state_.load(std::memory_order_relaxed) != Task::kLeader) {
    // wait for completion by group leader
    uint8_t desired_state = Task::kFinished | Task::kLeader;
    auto state = task->state_.load(std::memory_order_acquire);
    if ((state & desired_state) == 0 &&
        task->state_.compare_exchange_strong(state, Task::kWaiting)) {
      std::unique_lock<std::mutex> lock(task->state_mu_);
      task->state_cv_.wait(lock, [&task] {
        return task->state_.load(std::memory_order_relaxed) != Task::kWaiting;
      });
      state = task->state_.load(std::memory_order_relaxed);
    }
    if (state & Task::kFinished) {
      return;
    }
  }

  // is leader: build the group and execute
  assert(task->state_.load(std::memory_order_relaxed) == Task::kLeader);
  // if (task->prev_ != nullptr) {
  //   printf("leader task: %s (error prev not nullptr)\n",
  //   task->to_string().c_str());
  // }
  assert(task->prev_ == nullptr);

  auto group = create_group();
  group->SetLeader(task);
  group->ClaimTask(task);

  std::this_thread::sleep_for(std::chrono::microseconds(20));

  // this is the furthest task to claim
  newest_task = newest_task_.load(std::memory_order_relaxed);

  // backtrack to create all next links before claim
  BacktrackCreateNextLink(newest_task);
  Task* cand = task;
  while (cand != newest_task) {
    cand = cand->next_;
    if (!group->ClaimTask(cand)) {
      break;
    }
  }
  // printf("group size: %lu\n", group->size_);

  // sleep for a short period of time to allow other threads to claim the task

  // notify next leader
  auto last_task = group->last_task_;
  newest_task = newest_task_.load(std::memory_order_acquire);
  if (newest_task != last_task ||
      !newest_task_.compare_exchange_strong(newest_task, nullptr)) {
    BacktrackCreateNextLink(newest_task);
    // at least we should have one task after last task and by backtrack we
    // already establish the link
    assert(last_task->next_->prev_ == last_task);
    last_task->next_->prev_ = nullptr;
    auto next_leader = last_task->next_;

    auto next_leader_state =
        next_leader->state_.load(std::memory_order_acquire);
    if (next_leader_state == Task::kWaiting ||
        !next_leader->state_.compare_exchange_strong(next_leader_state,
                                                     Task::kLeader)) {
      assert(next_leader_state == Task::kWaiting);
      std::lock_guard<std::mutex> lock(next_leader->state_mu_);
      assert(next_leader->state_.load(std::memory_order_acquire) !=
             Task::kLeader);
      next_leader->state_.store(Task::kLeader, std::memory_order_relaxed);
      next_leader->state_cv_.notify_one();
    }
  }

  // should move to after notifying next leader, we shall allow multiple
  group->Run();

  // notify all the task in the group
  while (last_task != task) {
    auto prev_task = last_task->prev_;
    auto last_task_state = last_task->state_.load(std::memory_order_acquire);
    if (last_task_state == Task::kWaiting ||
        !last_task->state_.compare_exchange_strong(last_task_state,
                                                   Task::kFinished)) {
      assert(last_task_state == Task::kWaiting);
      std::lock_guard<std::mutex> lock(last_task->state_mu_);
      assert(last_task->state_.load(std::memory_order_acquire) !=
             Task::kFinished);
      last_task->state_.store(Task::kFinished, std::memory_order_relaxed);
      last_task->state_cv_.notify_one();
    }
    last_task = prev_task;
  }
}

}  // namespace hakes
