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

inline void parallelize(size_t begin, size_t end, size_t num_threads, size_t chunk_size,
                        std::function<void(size_t, size_t)> &&chunk_processor) {
  if (begin >= end)
    return;

  if (num_threads == 0)
    num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 1;

  if (chunk_size == 0)
    chunk_size = std::max<std::size_t>(1, (end - begin + num_threads - 1) / num_threads);

  std::atomic<size_t> next_begin{begin};
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    workers.emplace_back([&, chunk_size]() {
      while (true) {
        size_t start = next_begin.fetch_add(chunk_size, std::memory_order_relaxed);
        if (start >= end)
          break;

        size_t stop = std::min(end, start + chunk_size);
        chunk_processor(start, stop);
      }
    });
  }

  for (auto &worker : workers)
    worker.join();
}

} // namespace utils
