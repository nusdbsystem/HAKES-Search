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

#include "fnr-worker/worker.h"

#include <cassert>

namespace hakes {

FnRWorker::~FnRWorker() {
  if (engine_) {
    engine_->Close();
  }
}

bool FnRWorker::Initialize() { return engine_->IsInitialized(); }

bool FnRWorker::Handle(const std::string& url, const std::string& input,
                       std::string* output) {
  assert(engine_->IsInitialized());
  if (url == "/add") {
    return engine_->Add(input, output);
  } else if (url == "/search") {
    return engine_->Search(input, output);
  } else if (url == "/rerank") {
    return engine_->Rerank(input, output);
  } else if (url == "/checkpoint") {
    return engine_->Checkpoint();
  } else if (url == "/get_index") {
    return engine_->GetIndex(output);
  } else if (url == "/update_index") {
    return engine_->UpdateIndex(input, output);
  } else {
    return false;
  }
}

void FnRWorker::Close() {}

}  // namespace hakes