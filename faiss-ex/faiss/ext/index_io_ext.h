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

bool load_hakes_params(IOReader* f, HakesIndex* idx);
void save_hakes_params(IOWriter* f, const HakesIndex* idx);

bool load_hakes_findex(IOReader* ff, HakesIndex* idx);
bool load_hakes_rindex(IOReader* rf, HakesIndex* idx);
bool load_hakes_index(IOReader* ff, IOReader* rf, HakesIndex* idx, int mode);

void save_hakes_findex(IOWriter* ff, const HakesIndex* idx);

void save_hakes_rindex(IOWriter* rf, const HakesIndex* idx);

void save_hakes_uindex(IOWriter* uf, const HakesIndex* idx);

void save_hakes_index(IOWriter* ff, IOWriter* rf, const HakesIndex* idx);

// for now handled separatelyy
bool load_pa_map(IOReader* f, HakesIndex* idx);
void save_pa_map(IOWriter* f, const HakesIndex* idx);

void save_init_params(IOWriter* f, const std::vector<VectorTransform*>* vts,
                      ProductQuantizer* pq, IndexFlat* ivf);

}  // namespace faiss

#endif  // HAKES_INDEX_IO_EXT_H
