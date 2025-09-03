#include <catch2/catch_test_macros.hpp>
#include <dbscan_optimized.h>
#include <thread>
#include <vector>

TEST_CASE("AtomicUnionFind - Serial Operations", "[serial]") {
  SECTION("Initialization") {
    dbscan::AtomicUnionFind uf(10);
    for (int32_t i = 0; i < 10; ++i) {
      REQUIRE(uf.find(i) == i);
    }
  }

  SECTION("Simple Unite") {
    dbscan::AtomicUnionFind uf(10);
    uf.unite(0, 1);
    REQUIRE(uf.find(0) == uf.find(1));

    uf.unite(2, 3);
    REQUIRE(uf.find(2) == uf.find(3));

    REQUIRE(uf.find(0) != uf.find(2));
  }

  SECTION("Chain Unite") {
    dbscan::AtomicUnionFind uf(10);
    uf.unite(0, 1);
    uf.unite(1, 2);
    uf.unite(2, 3);

    int32_t root = uf.find(3); // Should be the root of the chain
    REQUIRE(uf.find(0) == root);
    REQUIRE(uf.find(1) == root);
    REQUIRE(uf.find(2) == root);
  }

  SECTION("Uniting Already United Sets") {
    dbscan::AtomicUnionFind uf(5);
    uf.unite(0, 1);
    uf.unite(2, 3);
    uf.unite(0, 3); // Unite the two sets

    int32_t root = uf.find(0);
    REQUIRE(uf.find(1) == root);
    REQUIRE(uf.find(2) == root);
    REQUIRE(uf.find(3) == root);

    // This should be a no-op and not cause issues
    uf.unite(1, 2);
    REQUIRE(uf.find(1) == root);
    REQUIRE(uf.find(2) == root);
  }

  SECTION("Multiple Unions") {
    dbscan::AtomicUnionFind uf(8);
    // Create two separate chains
    uf.unite(0, 1);
    uf.unite(1, 2);
    uf.unite(3, 4);
    uf.unite(4, 5);

    // Verify they're separate
    REQUIRE(uf.find(0) == uf.find(2));
    REQUIRE(uf.find(3) == uf.find(5));
    REQUIRE(uf.find(0) != uf.find(3));

    // Unite the chains
    uf.unite(2, 3);

    // Now they should be in the same set
    int32_t root = uf.find(0);
    REQUIRE(uf.find(1) == root);
    REQUIRE(uf.find(2) == root);
    REQUIRE(uf.find(3) == root);
    REQUIRE(uf.find(4) == root);
    REQUIRE(uf.find(5) == root);
  }
}

TEST_CASE("AtomicUnionFind - Concurrent Operations", "[concurrent]") {
  const int num_elements = 1000;
  const int num_threads = 16;
  dbscan::AtomicUnionFind uf(num_elements);
  std::vector<std::thread> threads;

  SECTION("Concurrent Disjoint Unite") {
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&, t]() {
        // Each thread works on its own slice of elements to avoid contention
        for (int i = t; i < num_elements / 2; i += num_threads) {
          uf.unite(2 * i, 2 * i + 1);
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }

    // Verification
    for (int i = 0; i < num_elements / 2; ++i) {
      REQUIRE(uf.find(2 * i) == uf.find(2 * i + 1));
      if (i > 0) {
        // Ensure disjoint sets remained disjoint
        REQUIRE(uf.find(2 * i) != uf.find(2 * (i - 1)));
      }
    }
  }

  SECTION("High Contention Unite (All to one root)") {
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&, t]() {
        // All threads try to unite their elements with element 0
        for (int i = t + 1; i < num_elements; i += num_threads) {
          uf.unite(0, i);
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }

    // Verification
    int32_t root = uf.find(0);
    for (int i = 1; i < num_elements; ++i) {
      REQUIRE(uf.find(i) == root);
    }
  }
}

TEST_CASE("AtomicUnionFind - Stress Test", "[concurrent][stress]") {
  const int num_elements = 2000;
  const int num_threads = std::thread::hardware_concurrency();
  dbscan::AtomicUnionFind uf(num_elements);
  std::vector<std::thread> threads;

  // In this test, all threads will concurrently unite even-indexed elements
  // into one set (rooted conceptually at 0) and odd-indexed elements into
  // another (rooted conceptually at 1). This creates contention while
  // having a predictable final state.

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      for (int i = t; i < num_elements; i += num_threads) {
        if (i > 1) {
          if (i % 2 == 0) {
            uf.unite(0, i); // Unite evens with 0
          } else {
            uf.unite(1, i); // Unite odds with 1
          }
        }
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  // Verification
  int32_t even_root = uf.find(0);
  int32_t odd_root = uf.find(1);

  REQUIRE(even_root != odd_root); // The two sets must be distinct

  for (int i = 0; i < num_elements; ++i) {
    if (i % 2 == 0) {
      REQUIRE(uf.find(i) == even_root);
    } else {
      REQUIRE(uf.find(i) == odd_root);
    }
  }
}

TEST_CASE("AtomicUnionFind - Edge Cases", "[edge_cases]") {
  SECTION("Single Element") {
    dbscan::AtomicUnionFind uf(1);
    REQUIRE(uf.find(0) == 0);
  }

  SECTION("Two Elements") {
    dbscan::AtomicUnionFind uf(2);
    REQUIRE(uf.find(0) == 0);
    REQUIRE(uf.find(1) == 1);

    uf.unite(0, 1);
    REQUIRE(uf.find(0) == uf.find(1));
  }

  SECTION("Self Unite") {
    dbscan::AtomicUnionFind uf(5);
    uf.unite(2, 2); // Should be a no-op
    REQUIRE(uf.find(2) == 2);
  }

  SECTION("Large Number of Elements") {
    const int large_n = 10000;
    dbscan::AtomicUnionFind uf(large_n);

    // Initialize check
    for (int i = 0; i < large_n; ++i) {
      REQUIRE(uf.find(i) == i);
    }

    // Create a chain
    for (int i = 0; i < large_n - 1; ++i) {
      uf.unite(i, i + 1);
    }

    // Verify all are in the same set
    int32_t root = uf.find(0);
    for (int i = 1; i < large_n; ++i) {
      REQUIRE(uf.find(i) == root);
    }
  }
}