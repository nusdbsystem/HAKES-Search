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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "fnr-worker/hakesbatchengine.h"
#include "fnr-worker/worker.h"
#include "llhttp.h"
#include "serving/server/server.h"
#include "serving/server/service.h"
#include "utils/fileutil.h"
#include "uv.h"

int main(int argc, char* argv[]) {
  // pick service selection

  if (argc < 5) {
    fprintf(stderr, "Usage: port data_path mode pa_mode\n");
    exit(1);
  }
  // parse the port
  auto port = std::stol(argv[1]);
  if (port < 0 || port > 65535) {
    fprintf(stderr, "Invalid port number\n");
    exit(1);
  }
  std::string path = argv[2];
  int mode = std::stoi(argv[3]);
  bool pa_mode = std::stoi(argv[4]);

  hakes::FnREngine* engine = new hakes::HakesBatchFnREngine();
  if (argc == 6) {
    uint64_t cap = std::stoll(argv[5]);
    if (cap > 0) {
      fprintf(stderr, "Invalid capacity %ld\n", cap);
      exit(1);
    }
    engine->Initialize(path, mode, pa_mode, cap);
  } else {
    engine->Initialize(path, mode, pa_mode);
  }

  hakes::Service s{std::unique_ptr<hakes::ServiceWorker>(
      new hakes::FnRWorker(std::unique_ptr<hakes::FnREngine>(engine)))};
  hakes::Server server(port, &s);
  if (!server.Init()) {
    fprintf(stderr, "Failed to initialize the server\n");
    exit(1);
  }
  printf("Service initialized\n");

  printf("Server starting\n");
  server.Start();

  return 0;
}

// ./sample-server 2351 path 0 0
