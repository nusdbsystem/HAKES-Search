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

#ifndef HAKES_INDEX_IO_EXT_H
#define HAKES_INDEX_IO_EXT_H

#include <faiss/IndexFlat.h>
#include <faiss/VectorTransform.h>
#include <faiss/ext/HakesIndex.h>
#include <faiss/ext/IdMap.h>
#include <faiss/impl/io.h>

namespace faiss {

struct Index;

const char ServingBaseIndexName[] = "base_index";
const char ServingPAMappingName[] = "pa_mapping";
const char ServingRefineIndexName[] = "refine_index";
const char ServingMappingName[] = "id_maps";
const char ServingServingConfigName[] = "serving_config";
const char HakesVTName[] = "pre-transform.bin";
const char HakesIVFName[] = "ivf.bin";
const char HakesPQName[] = "pq.bin";
const char HakesIVFVTName[] = "ivf_pre-transform.bin";

struct StringIOReader : IOReader {
  StringIOReader(const std::string& data) : data(data) {}
  std::string data;
  size_t rp = 0;
  size_t operator()(void* ptr, size_t size, size_t nitems) override;
};

struct StringIOWriter : IOWriter {
  std::string data;
  size_t operator()(const void* ptr, size_t size, size_t nitems) override;
};

// the io utilities are not thread safe, needs external synchronization

void write_index_ext(const Index* idx, const char* fname);
void write_index_ext(const Index* idx, IOWriter* f);

bool write_hakes_index(const char* fname, const Index* idx,
                       const std::vector<VectorTransform*>* vts = nullptr,
                       const std::vector<VectorTransform*>* ivf_vts = nullptr);

Index* read_index_ext(const char* fname, int io_flags = 0);
Index* read_index_ext(IOReader* f, int io_flags = 0);

Index* load_hakes_index(const char* fname, MetricType metric,
                        std::vector<VectorTransform*>* vts);
Index* load_hakes_index2(const char* fname, MetricType metric,
                         std::vector<VectorTransform*>* vts);

std::vector<idx_t> read_serving_config(const char* fname);
std::unordered_map<faiss::idx_t, faiss::idx_t> read_pa_mapping(const char* fname);
bool write_serving_config(const char* fname, const std::vector<idx_t>& config);
bool write_serving_index(
    const char* fname, const Index* base_idx,
    const std::vector<idx_t>& refine_scope, const Index* refine_idx,
    const IDMap& idmap, const std::vector<VectorTransform*>* vts = nullptr,
    const std::vector<VectorTransform*>* ivf_vts = nullptr,
    const std::unordered_map<faiss::idx_t, faiss::idx_t>& pa_mapping={});

bool write_hakes_vt_quantizers(const char* fname,
                               const std::vector<VectorTransform*>& pq_vts,
                               const std::vector<VectorTransform*>& ivf_vts,
                               const IndexFlat* ivf_centroids,
                               const ProductQuantizer* pq);

Index* load_hakes_vt_quantizers(const char* fname, MetricType metric,
                                std::vector<VectorTransform*>* pq_vts,
                                std::vector<VectorTransform*>* ivf_vts);
// single file read write out
bool write_hakes_vt_quantizers(IOWriter* f,
                               const std::vector<VectorTransform*>& pq_vts,
                               const IndexFlat* ivf_centroids,
                               const ProductQuantizer* pq) ;
Index* load_hakes_vt_quantizers(IOReader* f, MetricType metric,
                                std::vector<VectorTransform*>* pq_vts);

// parameters only
bool write_hakes_index_params(IOWriter* f,
                              const std::vector<VectorTransform*>& pq_vts,
                              const std::vector<VectorTransform*>& ivf_vts,
                              const IndexFlatL* ivf_centroids,
                              const ProductQuantizer* pq);

HakesIndex* load_hakes_index_params(IOReader* f);

bool load_hakes_index_single_file(IOReader* f, HakesIndex* idx);
bool write_hakes_index_single_file(IOWriter* f, const HakesIndex* idx);

}  // namespace faiss

#endif  // HAKES_INDEX_IO_EXT_H
