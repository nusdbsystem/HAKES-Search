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

#include "llhttp.h"
#include "serving/server/server.h"
#include "serving/server/service.h"
#include "shard-worker/hakesshardengine.h"
#include "shard-worker/worker.h"
#include "utils/fileutil.h"
#include "uv.h"

int main(int argc, char* argv[]) {
  // pick service selection

  if (argc != 3) {
    fprintf(stderr, "Usage: port data_path\n");
    exit(1);
  }
  // parse the port
  auto port = std::stol(argv[1]);
  if (port < 0 || port > 65535) {
    fprintf(stderr, "Invalid port number\n");
    exit(1);
  }
  std::string path = argv[2];

  hakes::ShardEngine* engine = new hakes::HakesShardEngine();
  engine->Initialize(path);

  hakes::Service s{std::unique_ptr<hakes::ServiceWorker>(
      new hakes::ShardWorker(std::unique_ptr<hakes::ShardEngine>(engine)))};
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

// ./sample-server 2351 path
