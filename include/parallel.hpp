#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

namespace utils {

inline void parallel_for(size_t begin, size_t end, size_t n_threads, const std::function<void(size_t, size_t)> &fn) {
  if (n_threads == 0)
    n_threads = std::thread::hardware_concurrency();
  if (n_threads == 0)
    n_threads = 1;

  size_t total = end - begin;
  size_t chunk = (total + n_threads - 1) / n_threads;

  std::vector<std::thread> threads;
  threads.reserve(n_threads);

  for (size_t t = 0; t < n_threads; ++t) {
    size_t chunk_begin = begin + t * chunk;
    if (chunk_begin >= end)
      break;
    size_t chunk_end = std::min(end, chunk_begin + chunk);

    threads.emplace_back([=]() { fn(chunk_begin, chunk_end); });
  }

  for (auto &th : threads)
    th.join();
}

} // namespace utils