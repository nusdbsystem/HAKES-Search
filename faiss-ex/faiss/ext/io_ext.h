#ifndef HAKES_IOEXT_H_
#define HAKES_IOEXT_H_

#include <faiss/impl/io.h>

namespace faiss {

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

}  // namespace faiss

#endif  // HAKES_IOEXT_H_
