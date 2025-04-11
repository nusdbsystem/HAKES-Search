#include <faiss/ext/io_ext.h>
#include <faiss/impl/io_macros.h>

#include <cassert>
#include <cstring>

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

}  // namespace faiss
