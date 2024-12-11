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

#ifndef HAKES_SERVING_TASK_QUEUE_TASK_H_
#define HAKES_SERVING_TASK_QUEUE_TASK_H_

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace hakes {

struct Task {
  enum TaskState : uint8_t {
    kInit = 0,
    kLeader = 1,
    kWaiting = 2,
    kFinished = 4,
  };

  virtual ~Task() {}
  virtual std::string to_string() {
    return "Task";
  }

  Task* prev_ = nullptr;
  Task* next_ = nullptr;
  std::atomic<uint8_t> state_{kInit};
  std::mutex state_mu_;
  std::condition_variable state_cv_;
};

struct TaskGroup {
  TaskGroup() : leader_(nullptr), last_task_(nullptr), size_(0) {}
  virtual ~TaskGroup() {}

  virtual void SetLeader(Task* leader) { leader_ = leader; }

  virtual bool ClaimTask(Task* task) {
    last_task_ = task;
    size_++;
    return true;
  }
  // run the task in group
  virtual bool Run() { return true; };

  Task* leader_ = nullptr;
  Task* last_task_ = nullptr;
  size_t size_;
};

struct TaskQueue {
  TaskQueue() : newest_task_(nullptr) {}
  virtual ~TaskQueue() {}

  virtual void Schedule(Task* task);

  virtual std::unique_ptr<TaskGroup> create_group() {
    return std::make_unique<TaskGroup>();
  }

  std::atomic<Task*> newest_task_;
};

}  // namespace hakes

#endif  // HAKES_SERVING_TASK_QUEUE_TASK_H_
