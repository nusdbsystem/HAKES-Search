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

#include <faiss/IndexFlat.h>
#include <faiss/IndexRefine.h>
#include <faiss/ext/BlockInvertedListsL.h>
#include <faiss/ext/IndexFlatL.h>
#include <faiss/ext/IndexIVFPQFastScanL.h>
#include <faiss/ext/IndexRefineL.h>
#include <faiss/ext/index_io_ext.h>
#include <faiss/ext/utils.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io_macros.h>
#include <faiss/index_io.h>

#include <filesystem>

namespace faiss {

size_t StringIOWriter::operator()(const void* ptr, size_t size, size_t nitems) {
  size_t bytes = size * nitems;
  if (bytes > 0) {
    size_t o = data.size();
    data.resize(o + bytes);
    memcpy(&data[o], ptr, size * nitems);
  }
  return nitems;
}

size_t StringIOReader::operator()(void* ptr, size_t size, size_t nitems) {
  if (rp >= data.size()) return 0;
  size_t nremain = (data.size() - rp) / size;
  if (nremain < nitems) nitems = nremain;
  if (size * nitems > 0) {
    memcpy(ptr, &data[rp], size * nitems);
    rp += size * nitems;
  }
  return nitems;
}

namespace {

void write_ivfl_header(const IndexIVFL* ivf, IOWriter* f) {
  write_index_header(ivf, f);
  WRITE1(ivf->nlist);
  WRITE1(ivf->nprobe);
  // subclasses write by_residual (some of them support only one setting of
  // by_residual).
  write_index_ext(ivf->quantizer, f);
  // write_direct_map(&ivf->direct_map, f);
}

void read_ivfl_header(IndexIVFL* ivf, IOReader* f,
                      std::vector<std::vector<idx_t>>* ids = nullptr) {
  read_index_header(ivf, f);
  READ1(ivf->nlist);
  READ1(ivf->nprobe);
  ivf->quantizer = read_index_ext(f);
  ivf->own_fields = true;
  if (ids) {  // used in legacy "Iv" formats
    ids->resize(ivf->nlist);
    for (size_t i = 0; i < ivf->nlist; i++) READVECTOR((*ids)[i]);
  }
}

void write_refine_map(const std::unordered_map<idx_t, idx_t>& m, IOWriter* f) {
  std::vector<std::pair<idx_t, idx_t>> v;
  v.resize(m.size());
  std::copy(m.begin(), m.end(), v.begin());
  WRITEVECTOR(v);
}

void read_refine_map(std::unordered_map<idx_t, idx_t>* m, IOReader* f) {
  std::vector<std::pair<idx_t, idx_t>> v;
  READVECTOR(v);
  m->clear();
  m->reserve(v.size());
  for (auto& p : v) {
    (*m)[p.first] = p.second;
  }
}

static void read_InvertedLists(IndexIVFL* ivf, IOReader* f, int io_flags) {
  InvertedLists* ils = read_InvertedLists(f, io_flags);
  if (ils) {
    FAISS_THROW_IF_NOT(ils->nlist == ivf->nlist);
    FAISS_THROW_IF_NOT(ils->code_size == InvertedLists::INVALID_CODE_SIZE ||
                       ils->code_size == ivf->code_size);
  }
  ivf->invlists = ils;
  ivf->own_invlists = true;
}
}  // anonymous namespace

void write_index_ext(const Index* idx, const char* fname) {
  FileIOWriter writer(fname);
  write_index_ext(idx, &writer);
}

void write_index_ext(const Index* idx, IOWriter* f) {
  register_bll_hook();  // register the BlockInvertedListsL hook.
  // check the new types before falling back to the original implementation
  if (const IndexFlatL* idxf = dynamic_cast<const IndexFlatL*>(idx)) {
    // same impl as IndexFlat, but with different fourcc for load
    uint32_t h = fourcc(idxf->metric_type == METRIC_INNER_PRODUCT ? "IlFI"
                        : idxf->metric_type == METRIC_L2          ? "IlF2"
                                                                  : "IlFl");
    WRITE1(h);
    write_index_header(idx, f);
    WRITEXBVECTOR(idxf->codes);
  } else if (const IndexRefineL* idxrf =
                 dynamic_cast<const IndexRefineL*>(idx)) {
    // Here we also need to store the mapping
    uint32_t h = fourcc("IlRF");
    WRITE1(h);
    // additionally store the two mapping
    write_refine_map(idxrf->off_to_idx, f);
    write_refine_map(idxrf->idx_to_off, f);

    write_index_header(idxrf, f);
    write_index_ext(idxrf->base_index, f);
    write_index_ext(idxrf->refine_index, f);
    WRITE1(idxrf->k_factor);
  } else if (const IndexIVFPQFastScanL* ivpq_2 =
                 dynamic_cast<const IndexIVFPQFastScanL*>(idx)) {
    // here we need to use the block inverted list locking IO
    uint32_t h = fourcc("IlPf");
    WRITE1(h);
    write_ivfl_header(ivpq_2, f);
    WRITE1(ivpq_2->by_residual);
    WRITE1(ivpq_2->code_size);
    WRITE1(ivpq_2->bbs);
    WRITE1(ivpq_2->M2);
    WRITE1(ivpq_2->implem);
    WRITE1(ivpq_2->qbs2);
    write_ProductQuantizer(&ivpq_2->pq, f);
    write_InvertedLists(ivpq_2->invlists, f);
  } else {
    write_index(idx, f);
  }
}

Index* read_index_ext(IOReader* f, int io_flags) {
  register_bll_hook();  // register the BlockInvertedListsL hook.
  Index* idx = nullptr;
  uint32_t h;
  READ1(h);
  if (h == fourcc("IlFI") || h == fourcc("IlF2") || h == fourcc("IlFl")) {
    IndexFlatL* idxf;
    if (h == fourcc("IlFI")) {
      idxf = new IndexFlatLIP();
    } else if (h == fourcc("IlF2")) {
      idxf = new IndexFlatLL2();
    } else {
      idxf = new IndexFlatL();
    }
    read_index_header(idxf, f);
    idxf->code_size = idxf->d * sizeof(float);
    READXBVECTOR(idxf->codes);
    FAISS_THROW_IF_NOT(idxf->codes.size() == idxf->ntotal * idxf->code_size);
    // leak!
    idx = idxf;
  } else if (h == fourcc("IlRF")) {
    IndexRefineL* idxrf = new IndexRefineL();
    read_refine_map(&idxrf->off_to_idx, f);
    read_refine_map(&idxrf->idx_to_off, f);

    read_index_header(idxrf, f);
    idxrf->base_index = read_index_ext(f, io_flags);
    // print memory after loading base index
    printf("Memory after loading base index: %ld\n",
           getCurrentRSS() / 1024 / 1024);
    idxrf->refine_index = read_index_ext(f, io_flags);
    READ1(idxrf->k_factor);
    if (dynamic_cast<IndexFlatL*>(idxrf->refine_index)) {
      // then make a RefineFlat with it
      IndexRefineL* idxrf_old = idxrf;
      idxrf = new IndexRefineFlatL();
      *idxrf = *idxrf_old;
      delete idxrf_old;
    }
    idxrf->own_fields = true;
    idxrf->own_refine_index = true;
    idx = idxrf;
    printf("Memory after loading refine index: %ld\n",
           getCurrentRSS() / 1024 / 1024);
  } else if (h == fourcc("IlPf")) {
    IndexIVFPQFastScanL* ivpq = new IndexIVFPQFastScanL();
    read_ivfl_header(ivpq, f);
    READ1(ivpq->by_residual);
    READ1(ivpq->code_size);
    READ1(ivpq->bbs);
    READ1(ivpq->M2);
    READ1(ivpq->implem);
    READ1(ivpq->qbs2);
    read_ProductQuantizer(&ivpq->pq, f);
    read_InvertedLists(ivpq, f, io_flags);
    ivpq->precompute_table();

    const auto& pq = ivpq->pq;
    ivpq->M = pq.M;
    ivpq->nbits = pq.nbits;
    ivpq->ksub = (1 << pq.nbits);
    ivpq->code_size = pq.code_size;
    printf("code_size: %ld\n", ivpq->code_size);
    ivpq->init_code_packer();

    idx = ivpq;
  } else {
    idx = read_index(f, io_flags);
  }
  return idx;
}

Index* read_index_ext(const char* fname, int io_flags) {
  FileIOReader reader(fname);
  return read_index_ext(&reader, io_flags);
}

bool read_hakes_pretransform(IOReader* f, std::vector<VectorTransform*>* vts) {
  // open pretransform file
  int32_t num_vt;
  READ1(num_vt);
  vts->reserve(num_vt);
  for (int i = 0; i < num_vt; i++) {
    int32_t d_out, d_in;
    READ1(d_out);
    READ1(d_in);
    size_t A_size = d_out * d_in;
    std::vector<float> A(d_out * d_in);
    READANDCHECK(A.data(), A_size);
    size_t b_size = d_out;
    std::vector<float> b(d_out);
    READANDCHECK(b.data(), b_size);
    LinearTransform* lt = new LinearTransform(d_in, d_out, true);
    lt->A = std::move(A);
    lt->b = std::move(b);
    lt->have_bias = true;
    lt->is_trained = true;
    vts->emplace_back(lt);
  }
  return true;
}

bool read_hakes_pretransform(const char* fname,
                             std::vector<VectorTransform*>* vts) {
  // open pretransform file
  FileIOReader reader(fname);
  return read_hakes_pretransform(&reader, vts);
}

IndexFlatL* read_hakes_ivf(IOReader* f, MetricType metric, bool* use_residual) {
  // open ivf file
  int32_t by_residual;
  READ1(by_residual);
  *use_residual = (by_residual == 1);
  int32_t nlist, d;
  READ1(nlist);
  READ1(d);
  IndexFlatL* ivf = new IndexFlatL(d, metric);
  size_t code_size = nlist * d * sizeof(float);
  std::vector<uint8_t> codes(code_size);
  READANDCHECK(codes.data(), code_size);
  printf("codes read size: %ld\n", code_size);
  ivf->codes = std::move(codes);
  ivf->is_trained = true;
  ivf->ntotal = nlist;
  return ivf;
}

IndexFlatL* read_hakes_ivf(const char* fname, MetricType metric,
                           bool* use_residual) {
  // open ivf file
  FileIOReader reader(fname);
  IOReader* f = &reader;
  return read_hakes_ivf(f, metric, use_residual);
}

bool read_hakes_pq(IOReader* f, ProductQuantizer* pq) {
  // open pq file
  int32_t d, M, nbits;
  READ1(d);
  READ1(M);
  READ1(nbits);
  pq->d = d;
  pq->M = M;
  pq->nbits = nbits;
  pq->set_derived_values();
  pq->train_type = ProductQuantizer::Train_hot_start;
  size_t centroids_size = pq->M * pq->ksub * pq->dsub;
  READANDCHECK(pq->centroids.data(), centroids_size);
  return true;
}

bool read_hakes_pq(const char* fname, ProductQuantizer* pq) {
  // open pq file
  FileIOReader reader(fname);
  return read_hakes_pq(&reader, pq);
}

Index* load_hakes_index(const char* fname, MetricType metric,
                        std::vector<VectorTransform*>* vts) {
  assert(vts != nullptr);
  // open pretransform file
  std::string vt_fname = std::string(fname) + "/" + HakesVTName;
  std::string ivf_fname = std::string(fname) + "/" + HakesIVFName;
  std::string pq_fname = std::string(fname) + "/" + HakesPQName;

  // load vts
  read_hakes_pretransform(vt_fname.c_str(), vts);

  // fast path if base_index exists
  std::string base_name = std::string(fname) + "/base_index";
  if (std::filesystem::exists(base_name)) {
    return read_index_ext(base_name.c_str());
  }

  // load quantizer
  bool use_residual;
  IndexFlatL* quantizer =
      read_hakes_ivf(ivf_fname.c_str(), metric, &use_residual);

  // load pq
  IndexIVFPQFastScanL* base_index = new IndexIVFPQFastScanL();
  read_hakes_pq(pq_fname.c_str(), &base_index->pq);
  // read_index_header
  base_index->d = quantizer->d;
  base_index->metric_type = metric;
  // read_ivfl_header
  base_index->nlist = quantizer->ntotal;
  base_index->quantizer = quantizer;
  base_index->own_fields = true;
  // read_index_ext IVFPQFastScanL branch
  base_index->by_residual = use_residual;
  base_index->code_size = base_index->pq.code_size;
  printf("code size: %ld\n", base_index->code_size);
  base_index->bbs = 32;
  base_index->M = base_index->pq.M;
  base_index->M2 = (base_index->M + 1) / 2 * 2;
  printf("M2: %ld\n", base_index->M2);
  base_index->implem = 0;
  base_index->qbs2 = 0;

  // read_InvertedLists
  CodePacker* code_packer = base_index->get_CodePacker();
  BlockInvertedListsL* il = new BlockInvertedListsL(
      base_index->nlist, code_packer->nvec, code_packer->block_size);
  il->init(nullptr, std::vector<int>());
  base_index->invlists = il;
  base_index->own_invlists = true;

  // base_index->precompute_table();
  base_index->nbits = base_index->pq.nbits;
  base_index->ksub = 1 << base_index->pq.nbits;
  base_index->code_size = base_index->pq.code_size;
  base_index->init_code_packer();

  base_index->is_trained = true;
  delete code_packer;
  return base_index;
}

IndexFlat* read_hakes_ivf2(IOReader* f, MetricType metric, bool* use_residual) {
  // open ivf file
  int32_t by_residual;
  READ1(by_residual);
  *use_residual = (by_residual == 1);
  int32_t nlist, d;
  READ1(nlist);
  READ1(d);
  IndexFlat* ivf = new IndexFlat(d, metric);
  size_t code_size = nlist * d * sizeof(float);
  std::vector<uint8_t> codes(code_size);
  READANDCHECK(codes.data(), code_size);
  printf("codes read size: %ld\n", code_size);
  ivf->codes = std::move(codes);
  ivf->is_trained = true;
  ivf->ntotal = nlist;
  return ivf;
}

IndexFlat* read_hakes_ivf2(const char* fname, MetricType metric,
                           bool* use_residual) {
  // open ivf file
  FileIOReader reader(fname);
  return read_hakes_ivf2(&reader, metric, use_residual);
}

Index* load_hakes_index2(const char* fname, MetricType metric,
                         std::vector<VectorTransform*>* vts) {
  assert(vts != nullptr);
  // open pretransform file
  std::string vt_fname = std::string(fname) + "/" + HakesVTName;
  std::string ivf_fname = std::string(fname) + "/" + HakesIVFName;
  std::string pq_fname = std::string(fname) + "/" + HakesPQName;

  // load vts
  read_hakes_pretransform(vt_fname.c_str(), vts);

  // load quantizer
  bool use_residual;

  IndexFlat* quantizer =
      read_hakes_ivf2(ivf_fname.c_str(), metric, &use_residual);

  // load pq
  IndexIVFPQ* base_index = new IndexIVFPQ();
  read_hakes_pq(pq_fname.c_str(), &base_index->pq);
  // assemble
  // IndexIVFL setting
  auto code_size = base_index->pq.code_size;
  base_index->code_size = code_size;
  base_index->by_residual = use_residual;
  // IndexIVFInterface no setting needed
  // Index setting
  base_index->d = quantizer->d;
  base_index->metric_type = metric;
  // quantizer setting
  base_index->quantizer = quantizer;
  base_index->nlist = quantizer->ntotal;
  base_index->invlists =
      new ArrayInvertedLists(base_index->nlist, base_index->code_size);
  base_index->invlists->code_size = base_index->pq.code_size;
  base_index->own_invlists = true;
  base_index->own_fields = true;

  base_index->precompute_table();
  base_index->is_trained = true;

  IndexRefineFlat* idxrf = new IndexRefineFlat(base_index);
  idxrf->own_fields = true;

  return idxrf;
}

bool write_hakes_pretransform(IOWriter* f,
                              const std::vector<VectorTransform*>* vts) {
  int32_t num_vt = vts->size();
  WRITE1(num_vt);
  for (int i = 0; i < num_vt; i++) {
    LinearTransform* lt = dynamic_cast<LinearTransform*>((*vts)[i]);
    if (lt == nullptr) {
      printf("write_hakes_pretransform: Only LinearTransform is supported\n");
      return false;
    }
    int32_t d_out = lt->d_out;
    int32_t d_in = lt->d_in;
    WRITE1(d_out);
    WRITE1(d_in);
    size_t A_size = d_out * d_in;
    WRITEANDCHECK(lt->A.data(), A_size);
    size_t b_size = d_out;
    if (!lt->have_bias) {
      auto zero_bias = std::vector<float>(d_out, 0);
      WRITEANDCHECK(zero_bias.data(), b_size);
    } else {
      WRITEANDCHECK(lt->b.data(), b_size);
    }
  }
  return true;
}

bool write_hakes_pretransform(const char* fname,
                              const std::vector<VectorTransform*>* vts) {
  FileIOWriter writer(fname);
  return write_hakes_pretransform(&writer, vts);
}

bool write_hakes_ivf(IOWriter* f, const Index* idx, bool use_residual) {
  const IndexFlatL* quantizer = dynamic_cast<const IndexFlatL*>(idx);
  if (quantizer == nullptr) {
    printf("write_hakes_ivf: Only IndexFlatL is supported\n");
    return false;
  }
  int32_t by_residual = use_residual ? 1 : 0;
  WRITE1(by_residual);
  int32_t nlist = quantizer->ntotal;
  int32_t d = quantizer->d;
  WRITE1(nlist);
  WRITE1(d);
  size_t code_size = nlist * d * sizeof(float);
  WRITEANDCHECK(quantizer->codes.data(), code_size);
  return true;
}

bool write_hakes_ivf(const char* fname, const Index* idx, bool use_residual) {
  FileIOWriter writer(fname);
  return write_hakes_ivf(&writer, idx, use_residual);
}

bool write_hakes_pq(IOWriter* f, const ProductQuantizer& pq) {
  int32_t d = pq.d;
  int32_t M = pq.M;
  int32_t nbits = pq.nbits;
  WRITE1(d);
  WRITE1(M);
  WRITE1(nbits);
  size_t centroids_size = M * pq.ksub * pq.dsub;
  WRITEANDCHECK(pq.centroids.data(), centroids_size);
  return true;
}

bool write_hakes_pq(const char* fname, const ProductQuantizer& pq) {
  FileIOWriter writer(fname);
  return write_hakes_pq(&writer, pq);
}

bool write_hakes_index(const char* fname, const Index* idx,
                       const std::vector<VectorTransform*>* vts,
                       const std::vector<VectorTransform*>* ivf_vts) {
  if (idx == nullptr) {
    printf("write_hakes_index: index is nullptr\n");
    return false;
  }

  // check and create directory
  std::filesystem::create_directories(fname);
  std::filesystem::permissions(fname,
                               std::filesystem::perms::owner_all |
                                   std::filesystem::perms::group_all |
                                   std::filesystem::perms::others_all,
                               std::filesystem::perm_options::add);

  // open pretransform file
  std::string vt_fname = std::string(fname) + "/" + HakesVTName;
  std::string ivf_fname = std::string(fname) + "/" + HakesIVFName;
  std::string pq_fname = std::string(fname) + "/" + HakesPQName;
  std::string ivf_vt_fname = std::string(fname) + "/" + HakesIVFVTName;

  // write vts
  if ((vts != nullptr) && (!write_hakes_pretransform(vt_fname.c_str(), vts))) {
    printf("write_hakes_index: write pretransform failed\n");
    return false;
  }

  // write ivf vts
  if (ivf_vts != nullptr) {
    if (!write_hakes_pretransform(ivf_vt_fname.c_str(), ivf_vts)) {
      printf("write_hakes_index: write ivf pretransform failed\n");
      return false;
    }
  }

  // write ivf
  const IndexIVFPQFastScanL* base_index =
      dynamic_cast<const IndexIVFPQFastScanL*>(idx);
  if (base_index == nullptr) {
    printf("write_hakes_index: Only IndexIVFPQFastScanL is supported\n");
    return false;
  }
  if (!write_hakes_ivf(ivf_fname.c_str(), base_index->quantizer,
                       base_index->by_residual)) {
    printf("write_hakes_index: write ivf failed\n");
    return false;
  }

  // write pq
  if (!write_hakes_pq(pq_fname.c_str(), base_index->pq)) {
    printf("write_hakes_index: write pq failed\n");
    return false;
  }

  // write the base index
  std::string base_name = std::string(fname) + "/base_index";
  write_index_ext(base_index, base_name.c_str());
  printf("write index success to %s\n", fname);
  return true;
}

bool write_serving_config(const char* fname,
                          const std::vector<idx_t>& refine_scope) {
  FileIOWriter writer(fname);
  IOWriter* f = &writer;
  uint32_t h = fourcc("Conf");
  WRITE1(h);
  WRITEVECTOR(refine_scope);
  return true;
}

std::vector<idx_t> read_serving_config(const char* fname) {
  // open config file
  FileIOReader reader(fname);
  IOReader* f = &reader;
  uint32_t h;
  READ1(h);
  if (h != fourcc("Conf")) {
    printf("read_serving_config: wrong file format\n");
    return std::vector<idx_t>{};
  }
  std::vector<idx_t> refine_scope;
  READVECTOR(refine_scope);
  return refine_scope;
}

std::unordered_map<faiss::idx_t, faiss::idx_t> read_pa_mapping(
    const char* fname) {
  // open pa mapping file
  FileIOReader reader(fname);
  IOReader* f = &reader;
  std::vector<std::pair<idx_t, idx_t>> v;
  READVECTOR(v);
  std::unordered_map<faiss::idx_t, faiss::idx_t> pa_mapping;
  pa_mapping.reserve(v.size());
  for (auto& p : v) {
    pa_mapping[p.first] = p.second;
  }
  return pa_mapping;
}

/**
 * @brief save the index to load in the serving framework
 *
 * @param fname dir to save index
 * @param base_idx base hakes index (vts, ivf centroids and pq codebook)
 * @param refine_scope the base index invlists index that the refine index
 * corresponds to
 * @param refine_idx refine index
 * @param idmap id mapping between refine index and base index
 * @param vts vector transformations
 * @return true
 * @return false
 */
bool write_serving_index(
    const char* fname, const Index* base_idx,
    const std::vector<idx_t>& refine_scope, const Index* refine_idx,
    const IDMap& idmap, const std::vector<VectorTransform*>* vts,
    const std::vector<VectorTransform*>* ivf_vts,
    const std::unordered_map<faiss::idx_t, faiss::idx_t>& pa_mapping) {
  // check and create directory
  std::filesystem::create_directories(fname);
  std::filesystem::permissions(fname,
                               std::filesystem::perms::owner_all |
                                   std::filesystem::perms::group_all |
                                   std::filesystem::perms::others_all,
                               std::filesystem::perm_options::add);
  std::string base_name = std::string(fname) + '/' + ServingBaseIndexName;
  std::string pa_mapping_name = std::string(fname) + '/' + ServingPAMappingName;
  std::string idmap_name = std::string(fname) + '/' + ServingMappingName;
  std::string refine_name = std::string(fname) + '/' + ServingRefineIndexName;
  std::string config_name = std::string(fname) + '/' + ServingServingConfigName;

  // write base index
  if (!write_hakes_index(base_name.c_str(), base_idx, vts)) {
    printf("write_serving_index: write base index failed\n");
    return false;
  }

  // write pa mapping
  if (!pa_mapping.empty()) {
    FileIOWriter writer(pa_mapping_name.c_str());
    FileIOWriter* f = &writer;
    std::vector<std::pair<idx_t, idx_t>> v;
    v.resize(pa_mapping.size());
    std::copy(pa_mapping.begin(), pa_mapping.end(), v.begin());
    WRITEVECTOR(v);
  }

  // write idmap
  if (!idmap.save(idmap_name.c_str())) {
    printf("write_serving_index: write idmap failed\n");
    return false;
  }

  // write refine index
  if (refine_idx != nullptr) {
    write_index_ext(refine_idx, refine_name.c_str());
  } else {
    printf("write_serving_index: refine index is nullptr, skipped\n");
  }

  // write config
  if (!write_serving_config(config_name.c_str(), refine_scope)) {
    printf("write_serving_index: write config failed\n");
    return false;
  }

  return true;
}

bool write_hakes_ivf2(IOWriter* f, const IndexFlat* idx, bool use_residual) {
  int32_t by_residual = use_residual ? 1 : 0;
  WRITE1(by_residual);
  int32_t nlist = idx->ntotal;
  int32_t d = idx->d;
  WRITE1(nlist);
  WRITE1(d);
  size_t code_size = nlist * d * sizeof(float);
  WRITEANDCHECK(idx->codes.data(), code_size);
  return true;
}

bool write_hakes_ivf2(const char* fname, const IndexFlat* idx,
                      bool use_residual) {
  FileIOWriter writer(fname);
  return write_hakes_ivf2(&writer, idx, use_residual);
}

bool write_hakes_vt_quantizers(const char* fname,
                               const std::vector<VectorTransform*>& pq_vts,
                               const std::vector<VectorTransform*>& ivf_vts,
                               const IndexFlat* ivf_centroids,
                               const ProductQuantizer* pq) {
  if ((ivf_centroids == nullptr) || (pq == nullptr)) {
    printf("write_hakes_vt_quantizers: ivf_centroids or pq is nullptr\n");
    return false;
  }

  // check and create directory
  std::filesystem::create_directories(fname);
  std::filesystem::permissions(fname,
                               std::filesystem::perms::owner_all |
                                   std::filesystem::perms::group_all |
                                   std::filesystem::perms::others_all,
                               std::filesystem::perm_options::add);

  // open pretransform file
  std::string pq_vt_fname = std::string(fname) + "/" + HakesVTName;
  std::string ivf_vt_fname = std::string(fname) + "/" + HakesIVFVTName;
  std::string ivf_fname = std::string(fname) + "/" + HakesIVFName;
  std::string pq_fname = std::string(fname) + "/" + HakesPQName;

  // write pq vts
  if (!write_hakes_pretransform(pq_vt_fname.c_str(), &pq_vts)) {
    printf("write_hakes_vt_quantizers: write pq vts failed\n");
    return false;
  }

  // write ivf vts
  if (ivf_vts.empty()) {
    printf(
        "write_hakes_vt_quantizers: ivf_vts is empty so skip (assume ivf use "
        "the same transform of opq)\n");
  } else {
    if (!write_hakes_pretransform(ivf_vt_fname.c_str(), &ivf_vts)) {
      printf("write_hakes_vt_quantizers: write ivf vts failed\n");
      return false;
    }
  }

  // write ivf
  if (!write_hakes_ivf2(ivf_fname.c_str(), ivf_centroids, false)) {
    printf("write_hakes_vt_quantizers: write ivf failed\n");
    return false;
  }

  // write pq
  if (!write_hakes_pq(pq_fname.c_str(), *pq)) {
    printf("write_hakes_vt_quantizers: write pq failed\n");
    return false;
  }

  return true;
}

Index* load_hakes_vt_quantizers(const char* fname, MetricType metric,
                                std::vector<VectorTransform*>* pq_vts,
                                std::vector<VectorTransform*>* ivf_vts) {
  assert(pq_vts != nullptr);
  assert(ivf_vts != nullptr);
  // open pretransform file
  std::string pq_vt_fname = std::string(fname) + "/" + HakesVTName;
  std::string ivf_vt_fname = std::string(fname) + "/" + HakesIVFVTName;
  std::string ivf_fname = std::string(fname) + "/" + HakesIVFName;
  std::string pq_fname = std::string(fname) + "/" + HakesPQName;

  // load pq vts
  read_hakes_pretransform(pq_vt_fname.c_str(), pq_vts);

  // load ivf vts
  // if no such file, skip
  if (!std::filesystem::exists(ivf_vt_fname)) {
    printf("load_hakes_vt_quantizers: ivf_vt_fname does not exist so skip\n");
  } else {
    read_hakes_pretransform(ivf_vt_fname.c_str(), ivf_vts);
  }

  // fast path if base_index exists
  std::string base_name = std::string(fname) + "/base_index";
  if (std::filesystem::exists(base_name)) {
    return read_index_ext(base_name.c_str());
  }

  // load ivf
  bool use_residual;
  IndexFlatL* quantizer =
      read_hakes_ivf(ivf_fname.c_str(), metric, &use_residual);

  // load pq
  IndexIVFPQFastScanL* base_index = new IndexIVFPQFastScanL();
  read_hakes_pq(pq_fname.c_str(), &base_index->pq);
  // read_index_header
  // use the opq vt to get the d
  base_index->d = pq_vts->back()->d_out;
  base_index->metric_type = metric;
  // read_ivfl_header
  base_index->nlist = quantizer->ntotal;
  base_index->quantizer = quantizer;
  base_index->own_fields = true;
  // read_index_ext IVFPQFastScanL branch
  base_index->by_residual = use_residual;
  base_index->code_size = base_index->pq.code_size;
  printf("code size: %ld\n", base_index->code_size);
  base_index->bbs = 32;
  base_index->M = base_index->pq.M;
  base_index->M2 = (base_index->M + 1) / 2 * 2;
  printf("M2: %ld\n", base_index->M2);
  base_index->implem = 0;
  base_index->qbs2 = 0;

  // read_InvertedLists
  CodePacker* code_packer = base_index->get_CodePacker();
  BlockInvertedListsL* il = new BlockInvertedListsL(
      base_index->nlist, code_packer->nvec, code_packer->block_size);
  il->init(nullptr, std::vector<int>());
  base_index->invlists = il;
  base_index->own_invlists = true;
  base_index->nbits = base_index->pq.nbits;
  base_index->ksub = 1 << base_index->pq.nbits;
  base_index->code_size = base_index->pq.code_size;
  base_index->init_code_packer();

  base_index->is_trained = true;

  delete code_packer;
  // return idxrf;
  return base_index;
}

// |---#vts----|---vts---|---ivf---|---pq---|
bool write_hakes_index_params(IOWriter* f,
                              const std::vector<VectorTransform*>& vts,
                              const std::vector<VectorTransform*>& ivf_vts,
                              const IndexFlatL* ivf_centroids,
                              const ProductQuantizer* pq) {
  // write vts
  if (!write_hakes_pretransform(f, &vts)) {
    printf("write_hakes_index_params: write pq vts failed\n");
    return false;
  }

  // write ivf vts
  if (!write_hakes_pretransform(f, &ivf_vts)) {
    printf("write_hakes_index_params: write ivf vts failed\n");
    return false;
  }

  // write ivf
  if (!write_hakes_ivf(f, ivf_centroids, false)) {
    printf("write_hakes_index_params: write ivf failed\n");
    return false;
  }

  // write pq
  if (!write_hakes_pq(f, *pq)) {
    printf("write_hakes_index_params: write pq failed\n");
    return false;
  }
  return true;
}

// many fields of the returned index is not initialized. just the parameters
HakesIndex* load_hakes_index_params(IOReader* f) {
  HakesIndex* index = new HakesIndex();

  // load pq vts
  read_hakes_pretransform(f, &index->vts_);

  // load ivf vts
  read_hakes_pretransform(f, &index->ivf_vts_);

  // load ivf
  bool use_residual;
  IndexFlatL* quantizer =
      read_hakes_ivf(f, METRIC_INNER_PRODUCT, &use_residual);

  // load pq

  // assemble
  IndexIVFPQFastScanL* base_index = new IndexIVFPQFastScanL();
  read_hakes_pq(f, &base_index->pq);
  base_index->d = index->vts_.back()->d_out;
  base_index->metric_type = METRIC_INNER_PRODUCT;
  base_index->nlist = quantizer->ntotal;
  base_index->quantizer = quantizer;
  base_index->own_fields = true;
  base_index->by_residual = use_residual;
  base_index->code_size = base_index->pq.code_size;
  base_index->bbs = 32;
  base_index->M = base_index->pq.M;
  base_index->M2 = (base_index->M + 1) / 2 * 2;
  base_index->implem = 0;
  base_index->qbs2 = 0;

  index->base_index_.reset(base_index);
  index->cq_ = index->base_index_->quantizer;
  return index;
}

// single file read write

bool write_hakes_vt_quantizers(IOWriter* f,
                               const std::vector<VectorTransform*>& pq_vts,
                               const IndexFlat* ivf_centroids,
                               const ProductQuantizer* pq) {
  if ((ivf_centroids == nullptr) || (pq == nullptr)) {
    printf("write_hakes_vt_quantizers: ivf_centroids or pq is nullptr\n");
    return false;
  }

  // write pq vts
  if (!write_hakes_pretransform(f, &pq_vts)) {
    printf("write_hakes_vt_quantizers: write pq vts failed\n");
    return false;
  }

  // write ivf
  if (!write_hakes_ivf2(f, ivf_centroids, false)) {
    printf("write_hakes_vt_quantizers: write ivf failed\n");
    return false;
  }

  // write pq
  if (!write_hakes_pq(f, *pq)) {
    printf("write_hakes_vt_quantizers: write pq failed\n");
    return false;
  }

  return true;
}

Index* load_hakes_vt_quantizers(IOReader* f, MetricType metric,
                                std::vector<VectorTransform*>* pq_vts) {
  assert(pq_vts != nullptr);
  read_hakes_pretransform(f, pq_vts);

  // load ivf
  bool use_residual;
  IndexFlatL* quantizer = read_hakes_ivf(f, metric, &use_residual);

  // load pq
  IndexIVFPQFastScanL* base_index = new IndexIVFPQFastScanL();
  read_hakes_pq(f, &base_index->pq);
  // read_index_header
  // use the opq vt to get the d
  base_index->d = pq_vts->back()->d_out;
  base_index->metric_type = metric;
  // read_ivfl_header
  base_index->nlist = quantizer->ntotal;
  base_index->quantizer = quantizer;
  base_index->own_fields = true;
  // read_index_ext IVFPQFastScanL branch
  base_index->by_residual = use_residual;
  base_index->code_size = base_index->pq.code_size;
  printf("code size: %ld\n", base_index->code_size);
  base_index->bbs = 32;
  base_index->M = base_index->pq.M;
  base_index->M2 = (base_index->M + 1) / 2 * 2;
  printf("M2: %ld\n", base_index->M2);
  base_index->implem = 0;
  base_index->qbs2 = 0;

  // read_InvertedLists
  CodePacker* code_packer = base_index->get_CodePacker();
  BlockInvertedListsL* il = new BlockInvertedListsL(
      base_index->nlist, code_packer->nvec, code_packer->block_size);
  il->init(nullptr, std::vector<int>());
  base_index->invlists = il;
  base_index->own_invlists = true;
  base_index->nbits = base_index->pq.nbits;
  base_index->ksub = 1 << base_index->pq.nbits;
  base_index->code_size = base_index->pq.code_size;
  base_index->init_code_packer();

  base_index->is_trained = true;

  delete code_packer;
  return base_index;
}

bool load_hakes_index_single_file(IOReader* f, HakesIndex* idx) {
  if (!read_hakes_pretransform(f, &idx->vts_)) {
    return false;
  }
  idx->base_index_.reset(
      dynamic_cast<faiss::IndexIVFPQFastScanL*>(read_index_ext(f)));
  idx->mapping_->load(f);
  idx->refine_index_.reset(dynamic_cast<faiss::IndexFlatL*>(read_index_ext(f)));
  return true;
}

bool write_hakes_index_single_file(IOWriter* f, const HakesIndex* idx) {
  if (!write_hakes_pretransform(f, &idx->vts_)) {
    return false;
  }
  write_index_ext(idx->base_index_.get(), f);
  if (!idx->mapping_->save(f)) {
    return false;
  }
  write_index_ext(idx->refine_index_.get(), f);
  return true;
}

}  // namespace faiss
