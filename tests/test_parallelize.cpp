#include "parallel.hpp"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <numeric>
#include <vector>

TEST_CASE("parallelize processes full range", "[parallel]") {
  constexpr std::size_t N = 10'000;
  std::vector<std::atomic<bool>> visited(N);
  for (auto &flag : visited)
    flag.store(false, std::memory_order_relaxed);

  utils::parallelize(0, N, 4, 128, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      bool expected = false;
      REQUIRE(visited[i].compare_exchange_strong(expected, true, std::memory_order_relaxed));
    }
  });

  for (const auto &flag : visited) {
    REQUIRE(flag.load(std::memory_order_relaxed));
  }
}

TEST_CASE("parallelize handles uneven chunks", "[parallel]") {
  constexpr std::size_t N = 1'023;
  std::vector<std::atomic<int>> counts(N);
  for (auto &value : counts)
    value.store(0, std::memory_order_relaxed);

  utils::parallelize(0, N, 3, 100, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      counts[i].fetch_add(1, std::memory_order_relaxed);
    }
  });

  for (const auto &value : counts) {
    REQUIRE(value.load(std::memory_order_relaxed) == 1);
  }
}

TEST_CASE("parallelize default chunk size", "[parallel]") {
  constexpr std::size_t N = 5'000;
  std::vector<int> out(N, 0);

  utils::parallelize(0, N, 8, 0, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      out[i] = static_cast<int>(i);
    }
  });

  for (std::size_t i = 0; i < N; ++i)
    REQUIRE(out[i] == static_cast<int>(i));
}
