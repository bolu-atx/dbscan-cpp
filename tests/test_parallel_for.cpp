#include "../include/parallel.hpp"
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <numeric>
#include <vector>

TEST_CASE("parallel_for basic functionality", "[parallel_for]") {
  const size_t n = 1000;
  std::vector<int> data(n, 0);

  utils::parallel_for(0, n, 4, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      data[i] = static_cast<int>(i);
    }
  });

  // Verify all elements are set correctly
  for (size_t i = 0; i < n; ++i) {
    REQUIRE(data[i] == static_cast<int>(i));
  }
}

TEST_CASE("parallel_for with zero threads", "[parallel_for]") {
  const size_t n = 100;
  std::vector<bool> processed(n, false);

  utils::parallel_for(0, n, 0, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      processed[i] = true;
    }
  });

  // Verify all elements are processed
  for (size_t i = 0; i < n; ++i) {
    REQUIRE(processed[i]);
  }
}

TEST_CASE("parallel_for with single thread", "[parallel_for]") {
  const size_t n = 50;
  std::vector<int> data(n);
  std::iota(data.begin(), data.end(), 0);

  int sum = 0;
  utils::parallel_for(0, n, 1, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      sum += data[i];
    }
  });

  int expected_sum = (n - 1) * n / 2;
  REQUIRE(sum == expected_sum);
}

TEST_CASE("parallel_for with empty range", "[parallel_for]") {
  bool called = false;
  utils::parallel_for(10, 10, 4, [&](size_t begin, size_t end) { called = true; });

  REQUIRE(!called);
}

TEST_CASE("parallel_for with single element", "[parallel_for]") {
  bool processed = false;
  utils::parallel_for(5, 6, 4, [&](size_t begin, size_t end) {
    REQUIRE(begin == 5);
    REQUIRE(end == 6);
    processed = true;
  });

  REQUIRE(processed);
}

TEST_CASE("parallel_for thread safety", "[parallel_for]") {
  const size_t n = 10000;
  std::atomic<int> counter(0);

  utils::parallel_for(0, n, 8, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      counter.fetch_add(1);
    }
  });

  REQUIRE(counter.load() == static_cast<int>(n));
}

TEST_CASE("parallel_for with more threads than elements", "[parallel_for]") {
  const size_t n = 3;
  std::vector<bool> processed(n, false);

  utils::parallel_for(0, n, 10, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      processed[i] = true;
    }
  });

  for (size_t i = 0; i < n; ++i) {
    REQUIRE(processed[i]);
  }
}

TEST_CASE("parallel_for with custom range", "[parallel_for]") {
  const size_t start = 100;
  const size_t end = 200;
  std::atomic<size_t> count(0);

  utils::parallel_for(start, end, 4, [&](size_t begin, size_t chunk_end) {
    for (size_t i = begin; i < chunk_end; ++i) {
      count.fetch_add(1);
    }
  });

  REQUIRE(count.load() == (end - start));
}
