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

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAKES_INDEX_REFINE_L_H_
#define HAKES_INDEX_REFINE_L_H_

#include <faiss/Index.h>

#include <shared_mutex>
#include <unordered_map>

namespace faiss {

struct IndexRefineLSearchParameters : SearchParameters {
  float k_factor = 1;
  SearchParameters* base_index_params = nullptr;  // non-owning

  virtual ~IndexRefineLSearchParameters() = default;
};

/** Index that queries in a base_index (a fast one) and refines the
 *  results with an exact search, hopefully improving the results.
 */
struct IndexRefineL : Index {
  /// faster index to pre-select the vectors that should be filtered
  Index* base_index;

  /// refinement index
  Index* refine_index;

  bool own_fields;        ///< should the base index be deallocated?
  bool own_refine_index;  ///< same with the refinement index

  /// factor between k requested in search and the k requested from
  /// the base_index (should be >= 1)
  float k_factor = 1;

  // add a map to map the idx to the offset
  std::unordered_map<idx_t, idx_t> idx_to_off;
  std::unordered_map<idx_t, idx_t> off_to_idx;

  mutable std::shared_mutex mu;

  bool get_is_trained() const override {
    std::shared_lock lock(mu);
    return is_trained;
  };

  idx_t get_ntotal() const override {
    std::shared_lock lock(mu);
    return ntotal;
  };

  /// initialize from empty index
  IndexRefineL(Index* base_index, Index* refine_index);

  IndexRefineL()
      : base_index(nullptr),
        refine_index(nullptr),
        own_fields(false),
        own_refine_index(false) {}

  // copy constructors
  IndexRefineL(const IndexRefineL& other) {
    // index fields
    d = other.d;
    ntotal = other.ntotal;
    verbose = other.verbose;
    is_trained = other.is_trained;
    metric_type = other.metric_type;
    metric_arg = other.metric_arg;
    // IndexRefineL fields
    base_index = other.base_index;
    refine_index = other.refine_index;
    own_fields = other.own_fields;
    own_refine_index = other.own_refine_index;
    k_factor = other.k_factor;
    idx_to_off = other.idx_to_off;
    off_to_idx = other.off_to_idx;
  };
  // copy assignment operator
  IndexRefineL& operator=(const IndexRefineL& other) {
    if (this == &other) {
      return *this;
    }
    // index fields
    d = other.d;
    ntotal = other.ntotal;
    verbose = other.verbose;
    is_trained = other.is_trained;
    metric_type = other.metric_type;
    metric_arg = other.metric_arg;
    // IndexRefineL fields
    base_index = other.base_index;
    refine_index = other.refine_index;
    own_fields = other.own_fields;
    own_refine_index = other.own_refine_index;
    k_factor = other.k_factor;
    idx_to_off = other.idx_to_off;
    off_to_idx = other.off_to_idx;
    return *this;
  };

  void train(idx_t n, const float* x) override;

  void add(idx_t n, const float* x) override;

  void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

  void reset() override;

  void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  // reconstruct is routed to the refine_index
  void reconstruct(idx_t key, float* recons) const override;

  /* standalone codec interface: the base_index codes are interleaved with the
   * refine_index ones */
  size_t sa_code_size() const override;

  void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

  /// The sa_decode decodes from the index_refine, which is assumed to be more
  /// accurate
  void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

  virtual void reserve(idx_t n) { /* noop */ }

  ~IndexRefineL() override;
};

/** Version where the refinement index is an IndexFlat. It has one additional
 * constructor that takes a table of elements to add to the flat refinement
 * index */
struct IndexRefineFlatL : IndexRefineL {
  explicit IndexRefineFlatL(Index* base_index);
  //   IndexRefineFlatL(Index* base_index, const float* xb);
  IndexRefineFlatL() : IndexRefineL() { own_refine_index = true; }

  void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
              const SearchParameters* params = nullptr) const override;

  void reserve(idx_t n) override;
};

}  // namespace faiss

#endif  // HAKES_INDEX_REFINE_L_H_