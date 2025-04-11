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

#include <faiss/ext/HakesIndex.h>
#include <faiss/ext/index_io_ext.h>

#include <iostream>

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << "D METRIC OUTPUT_PATH" << std::endl;
  }

  int d = std::stoi(argv[1]);
  int metric_type = std::stoi(argv[2]);
  faiss::MetricType metric =
      (metric_type == 1) ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
  std::string output_path = argv[3];
  std::cout << "d: " << d << ", metric: " << metric
            << ", output_path: " << output_path << std::endl;

  std::unique_ptr<faiss::FileIOWriter> rf =
      std::make_unique<faiss::FileIOWriter>(output_path.c_str());
  faiss::HakesIndex index;
  index.mapping_.reset(new faiss::IDMapImpl());
  index.refine_index_.reset(new faiss::IndexFlatL(d, metric));
  faiss::save_hakes_rindex(rf.get(), &index);
  std::cout << "Saved rindex to " << output_path << std::endl;
  return 0;
}
