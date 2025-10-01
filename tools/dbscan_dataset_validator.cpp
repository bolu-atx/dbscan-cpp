#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dbscan.h"
#include "dbscan_grid2d_l1.h"
#include "dbscan_optimized.h"

namespace {

struct Options {
  std::filesystem::path data_path{"data.bin"};
  std::filesystem::path truth_path{"truth.bin"};
  double eps{60.0};
  int32_t min_samples{16};

  bool run_baseline{true};
  bool run_optimized{true};
  bool run_grid_l1{false};
  std::optional<std::filesystem::path> mismatch_output_dir;
};

void print_usage(const char *program_name) {
  std::cout << "Usage: " << program_name
            << " [--data <data.bin>] [--truth <truth.bin>] [--eps <value>] [--min-samples <value>]"
            << " [--impl baseline|optimized|grid|both|all] [--dump-mismatches <directory>]\n";
}

Options parse_arguments(int argc, char **argv) {
  Options options;

  for (int i = 1; i < argc; ++i) {
    const std::string arg{argv[i]};

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--data") {
      if (i + 1 >= argc)
        throw std::invalid_argument("--data expects a path argument");
      options.data_path = argv[++i];
    } else if (arg == "--truth") {
      if (i + 1 >= argc)
        throw std::invalid_argument("--truth expects a path argument");
      options.truth_path = argv[++i];
    } else if (arg == "--eps") {
      if (i + 1 >= argc)
        throw std::invalid_argument("--eps expects a numeric argument");
      options.eps = std::stod(argv[++i]);
    } else if (arg == "--min-samples") {
      if (i + 1 >= argc)
        throw std::invalid_argument("--min-samples expects an integer argument");
      options.min_samples = static_cast<int32_t>(std::stoi(argv[++i]));
    } else if (arg == "--impl") {
      if (i + 1 >= argc)
        throw std::invalid_argument("--impl expects one of: baseline, optimized, grid, both, all");
      const std::string value{argv[++i]};
      if (value == "baseline") {
        options.run_baseline = true;
        options.run_optimized = false;
        options.run_grid_l1 = false;
      } else if (value == "optimized") {
        options.run_baseline = false;
        options.run_optimized = true;
        options.run_grid_l1 = false;
      } else if (value == "grid" || value == "grid_l1") {
        options.run_baseline = false;
        options.run_optimized = false;
        options.run_grid_l1 = true;
      } else if (value == "both") {
        options.run_baseline = true;
        options.run_optimized = true;
        options.run_grid_l1 = false;
      } else if (value == "all") {
        options.run_baseline = true;
        options.run_optimized = true;
        options.run_grid_l1 = true;
      } else {
        throw std::invalid_argument("--impl expects one of: baseline, optimized, grid, both, all");
      }
    } else if (arg == "--dump-mismatches") {
      if (i + 1 >= argc)
        throw std::invalid_argument("--dump-mismatches expects a directory path");
      options.mismatch_output_dir = std::filesystem::path{argv[++i]};
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  if (options.eps <= 0.0)
    throw std::invalid_argument("--eps must be positive");
  if (options.min_samples <= 0)
    throw std::invalid_argument("--min-samples must be positive");

  return options;
}

std::vector<dbscan::Point<double>> load_points(const std::filesystem::path &path, std::vector<uint32_t> *x_out,
                                               std::vector<uint32_t> *y_out) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Failed to open data file: " + path.string());

  stream.seekg(0, std::ios::end);
  const auto file_size = stream.tellg();
  stream.seekg(0, std::ios::beg);

  if (file_size % (sizeof(uint32_t) * 2) != 0)
    throw std::runtime_error("Data file does not contain a whole number of (y, x) uint32 pairs: " + path.string());

  const std::size_t num_values = static_cast<std::size_t>(file_size) / sizeof(uint32_t);
  const std::size_t num_points = num_values / 2;

  std::vector<uint32_t> raw(num_values);
  stream.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(uint32_t)));
  if (!stream)
    throw std::runtime_error("Failed to read data file: " + path.string());

  std::vector<dbscan::Point<double>> points(num_points);
  if (x_out)
    x_out->assign(num_points, 0U);
  if (y_out)
    y_out->assign(num_points, 0U);

  for (std::size_t i = 0; i < num_points; ++i) {
    const uint32_t y = raw[2 * i];
    const uint32_t x = raw[2 * i + 1];
    points[i] = dbscan::Point<double>{static_cast<double>(x), static_cast<double>(y)};
    if (x_out)
      (*x_out)[i] = x;
    if (y_out)
      (*y_out)[i] = y;
  }

  return points;
}

std::vector<int32_t> load_labels(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Failed to open truth file: " + path.string());

  stream.seekg(0, std::ios::end);
  const auto file_size = stream.tellg();
  stream.seekg(0, std::ios::beg);

  if (file_size % sizeof(int32_t) != 0)
    throw std::runtime_error("Truth file does not contain a whole number of int32 labels: " + path.string());

  const std::size_t num_labels = static_cast<std::size_t>(file_size) / sizeof(int32_t);
  std::vector<int32_t> labels(num_labels);
  stream.read(reinterpret_cast<char *>(labels.data()), static_cast<std::streamsize>(labels.size() * sizeof(int32_t)));
  if (!stream)
    throw std::runtime_error("Failed to read truth file: " + path.string());

  return labels;
}

std::size_t count_clusters(const std::vector<int32_t> &labels) {
  std::unordered_set<int32_t> clusters;
  clusters.reserve(labels.size());
  for (int32_t label : labels) {
    if (label != -1)
      clusters.insert(label);
  }
  return clusters.size();
}

std::size_t count_noise(const std::vector<int32_t> &labels) {
  return static_cast<std::size_t>(std::count(labels.begin(), labels.end(), -1));
}

struct LabelIndex {
  std::unordered_map<int32_t, std::size_t> to_index;
  std::vector<int32_t> values;
};

LabelIndex make_index(const std::vector<int32_t> &labels) {
  LabelIndex index;
  index.values.reserve(labels.size());
  index.to_index.reserve(labels.size());

  for (int32_t label : labels) {
    if (!index.to_index.contains(label)) {
      const std::size_t idx = index.values.size();
      index.values.push_back(label);
      index.to_index.emplace(label, idx);
    }
  }

  return index;
}

double combination2(std::int64_t n) {
  if (n <= 1)
    return 0.0;
  return static_cast<double>(n) * static_cast<double>(n - 1) / 2.0;
}

struct EvaluationMetrics {
  double adjusted_rand{0.0};
  double remapped_accuracy{0.0};
  std::size_t mismatched_points{0};
  std::size_t predicted_clusters{0};
  std::size_t truth_clusters{0};
  std::size_t predicted_noise{0};
  std::size_t truth_noise{0};
  bool passed{false};
};

EvaluationMetrics evaluate(const std::vector<int32_t> &predicted, const std::vector<int32_t> &truth,
                           std::vector<std::size_t> *mismatch_indices = nullptr) {
  if (predicted.size() != truth.size())
    throw std::runtime_error("Predicted labels and truth labels must have the same length");

  const std::size_t total_points = truth.size();
  const auto predicted_index = make_index(predicted);
  const auto truth_index = make_index(truth);

  const std::size_t predicted_size = predicted_index.values.size();
  const std::size_t truth_size = truth_index.values.size();

  std::vector<std::int64_t> contingency(predicted_size * truth_size, 0);
  std::vector<std::int64_t> predicted_counts(predicted_size, 0);
  std::vector<std::int64_t> truth_counts(truth_size, 0);

  for (std::size_t i = 0; i < total_points; ++i) {
    const auto predicted_it = predicted_index.to_index.find(predicted[i]);
    const auto truth_it = truth_index.to_index.find(truth[i]);
    const std::size_t predicted_row = predicted_it->second;
    const std::size_t truth_col = truth_it->second;

    const std::size_t cell_index = predicted_row * truth_size + truth_col;
    ++contingency[cell_index];
    ++predicted_counts[predicted_row];
    ++truth_counts[truth_col];
  }

  double sum_combination = 0.0;
  for (std::int64_t count : contingency)
    sum_combination += combination2(count);

  double predicted_combination = 0.0;
  for (std::int64_t count : predicted_counts)
    predicted_combination += combination2(count);

  double truth_combination = 0.0;
  for (std::int64_t count : truth_counts)
    truth_combination += combination2(count);

  const double total_pairs = combination2(static_cast<std::int64_t>(total_points));
  double expected_index = 0.0;
  if (total_pairs > 0.0)
    expected_index = (predicted_combination * truth_combination) / total_pairs;

  const double max_index = 0.5 * (predicted_combination + truth_combination);
  const double denominator = max_index - expected_index;

  EvaluationMetrics metrics;
  if (denominator == 0.0) {
    metrics.adjusted_rand = 1.0;
  } else {
    metrics.adjusted_rand = (sum_combination - expected_index) / denominator;
  }

  std::unordered_map<int32_t, int32_t> remap;
  remap.reserve(predicted_size);
  for (std::size_t row = 0; row < predicted_size; ++row) {
    const int32_t predicted_label = predicted_index.values[row];
    if (predicted_label == -1) {
      remap.emplace(predicted_label, -1);
      continue;
    }

    const std::int64_t *row_ptr = contingency.data() + row * truth_size;
    std::size_t best_col = 0;
    std::int64_t best_count = -1;
    for (std::size_t col = 0; col < truth_size; ++col) {
      if (row_ptr[col] > best_count) {
        best_count = row_ptr[col];
        best_col = col;
      }
    }
    remap.emplace(predicted_label, truth_index.values[best_col]);
  }

  if (mismatch_indices)
    mismatch_indices->clear();

  std::size_t matches = 0;
  for (std::size_t i = 0; i < total_points; ++i) {
    const int32_t predicted_label = predicted[i];
    const auto mapping_it = remap.find(predicted_label);
    const int32_t mapped_label = mapping_it != remap.end() ? mapping_it->second : predicted_label;
    if (mapped_label == truth[i])
      ++matches;
    else if (mismatch_indices)
      mismatch_indices->push_back(i);
  }

  metrics.remapped_accuracy =
      total_points == 0 ? 1.0 : static_cast<double>(matches) / static_cast<double>(total_points);
  metrics.mismatched_points = mismatch_indices ? mismatch_indices->size() : (total_points - matches);

  metrics.predicted_clusters = count_clusters(predicted);
  metrics.truth_clusters = count_clusters(truth);
  metrics.predicted_noise = count_noise(predicted);
  metrics.truth_noise = count_noise(truth);
  metrics.passed = metrics.mismatched_points == 0 && metrics.predicted_clusters == metrics.truth_clusters;

  return metrics;
}

struct RunResult {
  std::string name;
  EvaluationMetrics metrics;
};

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_arguments(argc, argv);

    std::vector<uint32_t> x_coords;
    std::vector<uint32_t> y_coords;
    const auto points = load_points(options.data_path, &x_coords, &y_coords);
    const auto truth_labels = load_labels(options.truth_path);

    if (points.size() != truth_labels.size())
      throw std::runtime_error("Point count and truth label count differ");

    std::cout << "Loaded " << points.size() << " points from " << options.data_path << "\n";
    std::cout << "Using eps=" << options.eps << ", min_samples=" << options.min_samples << "\n";

    const auto truth_cluster_count = count_clusters(truth_labels);
    const auto truth_noise_count = count_noise(truth_labels);
    std::cout << "Ground truth clusters: " << truth_cluster_count << "; noise points: " << truth_noise_count << "\n";

    std::vector<RunResult> results;
    results.reserve(3);

    if (options.run_baseline) {
      std::cout << "\n[baseline] Running clustering..." << std::flush;
      const auto start = std::chrono::steady_clock::now();
      dbscan::DBSCAN<double> baseline(options.eps, options.min_samples);
      const auto clustering = baseline.cluster(points);
      std::vector<std::size_t> mismatches;
      const auto metrics =
          evaluate(clustering.labels, truth_labels, options.mismatch_output_dir ? &mismatches : nullptr);
      const auto end = std::chrono::steady_clock::now();
      const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      std::cout << " done in " << elapsed_ms << " ms" << std::endl;
      results.push_back({"baseline", metrics});

      if (options.mismatch_output_dir && !mismatches.empty()) {
        std::filesystem::create_directories(*options.mismatch_output_dir);
        auto file_path = *options.mismatch_output_dir / "baseline_mismatches.txt";
        std::ofstream out(file_path);
        if (!out)
          throw std::runtime_error("Failed to open mismatch output file: " + file_path.string());
        for (std::size_t index : mismatches)
          out << index << '\n';
        std::cout << "[baseline] Wrote " << mismatches.size() << " mismatches to " << file_path << "\n";
      }
    }

    if (options.run_optimized) {
      std::cout << "\n[optimized] Running clustering..." << std::flush;
      const auto start = std::chrono::steady_clock::now();
      dbscan::DBSCANOptimized<double> optimized(options.eps, options.min_samples);
      const auto clustering = optimized.cluster(points);
      std::vector<std::size_t> mismatches;
      const auto metrics =
          evaluate(clustering.labels, truth_labels, options.mismatch_output_dir ? &mismatches : nullptr);
      const auto end = std::chrono::steady_clock::now();
      const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      std::cout << " done in " << elapsed_ms << " ms" << std::endl;
      results.push_back({"optimized", metrics});

      if (options.mismatch_output_dir && !mismatches.empty()) {
        std::filesystem::create_directories(*options.mismatch_output_dir);
        auto file_path = *options.mismatch_output_dir / "optimized_mismatches.txt";
        std::ofstream out(file_path);
        if (!out)
          throw std::runtime_error("Failed to open mismatch output file: " + file_path.string());
        for (std::size_t index : mismatches)
          out << index << '\n';
        std::cout << "[optimized] Wrote " << mismatches.size() << " mismatches to " << file_path << "\n";
      }
    }

    if (options.run_grid_l1) {
      const auto eps_int = static_cast<uint32_t>(std::llround(options.eps));
      if (std::fabs(options.eps - static_cast<double>(eps_int)) > 1e-6) {
        throw std::invalid_argument("grid_l1 implementation requires integer eps value");
      }
      if (x_coords.size() != y_coords.size())
        throw std::runtime_error("Mismatch between x and y coordinate counts");

      std::cout << "\n[grid_l1] Running clustering..." << std::flush;
      const auto start = std::chrono::steady_clock::now();
      dbscan::DBSCANGrid2D_L1 grid_algo(eps_int, static_cast<uint32_t>(options.min_samples));
      const auto labels = grid_algo.fit_predict(x_coords.data(), y_coords.data(), x_coords.size());
      std::vector<std::size_t> mismatches;
      const auto metrics = evaluate(labels, truth_labels, options.mismatch_output_dir ? &mismatches : nullptr);
      const auto end = std::chrono::steady_clock::now();
      const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      std::cout << " done in " << elapsed_ms << " ms" << std::endl;
      results.push_back({"grid_l1", metrics});

      if (options.mismatch_output_dir && !mismatches.empty()) {
        std::filesystem::create_directories(*options.mismatch_output_dir);
        auto file_path = *options.mismatch_output_dir / "grid_l1_mismatches.txt";
        std::ofstream out(file_path);
        if (!out)
          throw std::runtime_error("Failed to open mismatch output file: " + file_path.string());
        for (std::size_t index : mismatches)
          out << index << '\n';
        std::cout << "[grid_l1] Wrote " << mismatches.size() << " mismatches to " << file_path << "\n";
      }
    }

    std::cout << std::fixed << std::setprecision(6);

    bool all_passed = true;
    for (const auto &result : results) {
      const auto &m = result.metrics;
      std::cout << "\nImplementation: " << result.name << "\n";
      std::cout << "  clusters: " << m.predicted_clusters << " (truth " << m.truth_clusters << ")\n";
      std::cout << "  noise points: " << m.predicted_noise << " (truth " << m.truth_noise << ")\n";
      std::cout << "  adjusted rand index: " << m.adjusted_rand << "\n";
      std::cout << "  remapped accuracy: " << m.remapped_accuracy * 100.0 << "%\n";
      std::cout << "  mismatched points: " << m.mismatched_points << "\n";
      std::cout << "  status: " << (m.passed ? "PASS" : "FAIL") << "\n";
      all_passed = all_passed && m.passed;
    }

    return all_passed ? 0 : 1;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    print_usage(argv[0]);
    return 1;
  }
}
