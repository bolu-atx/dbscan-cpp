#pragma once

#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace dbscan {

struct PerfTimingEntry {
  std::string label;
  double duration_ms;
};

struct PerfTiming {
  void clear() { entries_.clear(); }

  void add(std::string label, double duration_ms) {
    entries_.push_back({std::move(label), duration_ms});
  }

  [[nodiscard]] const std::vector<PerfTimingEntry> &entries() const { return entries_; }

private:
  std::vector<PerfTimingEntry> entries_;

  friend class ScopedTimer;
};

class ScopedTimer {
public:
  ScopedTimer(std::string label, PerfTiming &sink)
      : sink_(sink), label_(std::move(label)), start_(std::chrono::steady_clock::now()) {}

  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;

  ~ScopedTimer() {
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start_);
    sink_.add(std::move(label_), elapsed.count());
  }

private:
  PerfTiming &sink_;
  std::string label_;
  std::chrono::steady_clock::time_point start_;
};

} // namespace dbscan

