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

#include <faiss/ext/BlockInvertedListsL.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>

#include <mutex>

namespace {
std::once_flag register_bll_hook_once;
}  // anonymous namespace

namespace faiss {

size_t InvListUnit::resize(size_t new_size) {
  std::unique_lock lock(mu_);
  return resize_nolock(new_size);
}

size_t InvListUnit::resize_nolock(size_t new_size) {
  ids_.resize(new_size);
  size_t prev_nbytes = codes_.size();
  size_t n_block = (new_size + n_per_block - 1) / n_per_block;
  size_t new_nbytes = n_block * block_size;
  codes_.resize(new_nbytes);
  // set new bytes to 0
  if (prev_nbytes < codes_.size()) {
    memset(codes_.data() + prev_nbytes, 0, codes_.size() - prev_nbytes);
  }
  count_.store(new_size, std::memory_order_relaxed);
  return new_size;
}

// 0 for init_list_cap means keep the current codes and ids
size_t InvListUnit::set_active(size_t n_per_block, size_t block_size,
                               size_t init_list_cap) {
  this->n_per_block = n_per_block;
  this->block_size = block_size;
  this->active = true;
  if (init_list_cap == 0) {
    return 0;
  }
  this->codes_.resize(init_list_cap);
  this->ids_.resize(init_list_cap);
  memset(this->codes_.data(), 0, this->codes_.size());
  return 0;
}

size_t InvListUnit::add_entries(size_t n_entry, const idx_t* ids_in,
                                const uint8_t* code, const CodePacker* packer) {
  if (n_entry == 0) {
    return 0;
  }
  // lock the list
  std::unique_lock lock(this->mu_);
  // add ids
  size_t o = this->ids_.size();
  this->ids_.resize(o + n_entry);
  memcpy(&this->ids_[o], ids_in, sizeof(ids_in[0]) * n_entry);
  size_t n_block = (o + n_entry + n_per_block - 1) / n_per_block;
  // add codes
  this->codes_.resize(n_block * block_size);
  if (o % block_size == 0) {
    memcpy(&this->codes_[o * block_size], code, n_block * block_size);
  } else {
    FAISS_THROW_IF_NOT_MSG(packer, "missing code packer");
    std::vector<uint8_t> buffer(packer->code_size);
    for (size_t i = 0; i < n_entry; i++) {
      packer->unpack_1(code, i, buffer.data());
      packer->pack_1(buffer.data(), i + o, this->codes_.data());
    }
  }
  return o;
}

const uint8_t* InvListUnit::get_codes() const {
  this->mu_.lock_shared();
  return this->codes_.data();
};

const idx_t* InvListUnit::get_ids() const {
  this->mu_.lock_shared();
  return this->ids_.data();
};

void InvListUnit::release_codes(const uint8_t* codes) const {
  this->mu_.unlock_shared();
};

void InvListUnit::release_ids(const idx_t* ids) const {
  this->mu_.unlock_shared();
};

void register_bll_hook() {
  std::call_once(register_bll_hook_once, []() {
    InvertedListsIOHook::add_callback(new BlockInvertedListsLIOHook());
  });
}

BlockInvertedListsL::BlockInvertedListsL(size_t nlist, size_t vec_per_block,
                                         size_t block_size)
    : InvertedLists(nlist, InvertedLists::INVALID_CODE_SIZE),
      n_per_block_(vec_per_block),
      block_size_(block_size) {
  lists_.resize(nlist);
}

void BlockInvertedListsL::init(const CodePacker* packer,
                               const std::vector<int>& load_list,
                               size_t init_list_cap) {
  // this->packer_factory_ = packer_factory;
  if (this->packer_) {
    delete this->packer_;
  }
  this->packer_ = packer;
  this->load_list_ = load_list;
  if (this->load_list_.empty()) {
    // load all the list
    this->load_list_.resize(nlist);
    for (size_t i = 0; i < nlist; i++) {
      this->load_list_[i] = i;
    }
  }
  for (auto idx : this->load_list_) {
    this->lists_[idx].set_active(n_per_block_, block_size_, init_list_cap);
  }
}

void BlockInvertedListsL::set_code_packer(const CodePacker* packer) {
  if (this->packer_) {
    delete this->packer_;
  }
  this->packer_ = packer;
  assert(this->packer_->block_size == block_size_);
  assert(this->packer_->nvec == n_per_block_);
}

size_t BlockInvertedListsL::add_entries(size_t list_no, size_t n_entry,
                                        const idx_t* ids_in,
                                        const uint8_t* code) {
  if (n_entry == 0) {
    return 0;
  }
  FAISS_THROW_IF_NOT(list_no < nlist);
  return this->lists_[list_no].add_entries(n_entry, ids_in, code, packer_);
}

size_t BlockInvertedListsL::list_size(size_t list_no) const {
  assert(list_no < this->nlist);
  return this->lists_[list_no].get_size();
}

// not really used
const uint8_t* BlockInvertedListsL::get_codes(size_t list_no) const {
  assert(list_no < nlist);
  return this->lists_[list_no].get_codes();
}

const idx_t* BlockInvertedListsL::get_ids(size_t list_no) const {
  assert(list_no < nlist);
  return this->lists_[list_no].get_ids();
}

void BlockInvertedListsL::release_codes(size_t list_no,
                                        const uint8_t* codes) const {
  assert(list_no < nlist);
  return this->lists_[list_no].release_codes(codes);
}

void BlockInvertedListsL::release_ids(size_t list_no, const idx_t* ids) const {
  assert(list_no < nlist);
  return this->lists_[list_no].release_ids(ids);
}

const uint8_t* BlockInvertedListsL::get_single_code(size_t list_no,
                                                    size_t offset) const {
  FAISS_THROW_MSG("get single code not implemented");
}

// not used
void BlockInvertedListsL::resize(size_t list_no, size_t new_size) {
  this->lists_[list_no].resize(new_size);
}

void BlockInvertedListsL::resize_nolock(size_t list_no, size_t new_size) {
  this->lists_[list_no].resize_nolock(new_size);
}

void BlockInvertedListsL::update_entries(size_t, size_t, size_t, const idx_t*,
                                         const uint8_t*) {
  FAISS_THROW_MSG("not impemented");
}

BlockInvertedListsL::~BlockInvertedListsL() {
  if (packer_) {
    delete packer_;
  }
}

/**************************************************
 * IO hook implementation
 **************************************************/

BlockInvertedListsLIOHook::BlockInvertedListsLIOHook()
    : InvertedListsIOHook("ibll", typeid(BlockInvertedListsL).name()) {
}

void BlockInvertedListsLIOHook::write(const InvertedLists* ils_in,
                                      IOWriter* f) const {
  uint32_t h = fourcc("ibll");
  WRITE1(h);
  const BlockInvertedListsL* il =
      dynamic_cast<const BlockInvertedListsL*>(ils_in);
  WRITE1(il->nlist);
  WRITE1(il->code_size);
  WRITE1(il->n_per_block_);
  WRITE1(il->block_size_);

  // write the load list
  WRITEVECTOR(il->load_list_);

  for (size_t i = 0; i < il->nlist; i++) {
    il->lists_[i].write(f);
  }
}

void InvListUnit::write(IOWriter* f) const {
  WRITEVECTOR(ids_);
  WRITEVECTOR(codes_);
}

void InvListUnit::read(IOReader* f) {
  READVECTOR(ids_);
  READVECTOR(codes_);
  count_.store(ids_.size(), std::memory_order_relaxed);
}

InvertedLists* BlockInvertedListsLIOHook::read(IOReader* f,
                                               int /* io_flags */) const {

  size_t nlist, code_size, n_per_block, block_size;
  std::vector<int> load_list;
  READ1(nlist);
  READ1(code_size);
  READ1(n_per_block);
  READ1(block_size);
  READVECTOR(load_list);

  BlockInvertedListsL* il =
      new BlockInvertedListsL(nlist, n_per_block, block_size);

  for (size_t i = 0; i < il->nlist; i++) {
    il->lists_[i].read(f);
  }

  // set load list and active, capacity is 0 to keep current loaded lists
  il->init(nullptr, load_list);

  return il;
}

}  // namespace faiss
