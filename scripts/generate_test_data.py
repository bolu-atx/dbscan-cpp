#!/usr/bin/env -S uv run
# /// script
# dependencies = ["numpy", "scikit-learn"]
# ///

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic DBSCAN test data with embedded Gaussian clusters."
    )
    parser.add_argument(
        "--uniform-count",
        type=int,
        default=200_000,
        help="Number of background points drawn from a uniform distribution (default: 200_000).",
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=100,
        help="Number of Gaussian clusters to sprinkle into the dataset (default: 100).",
    )
    parser.add_argument(
        "--points-per-cluster",
        type=int,
        default=256,
        help="Number of points sampled for each Gaussian cluster (default: 256).",
    )
    parser.add_argument(
        "--area-width",
        type=int,
        default=1_000_000,
        help="Width/height of the square area measured in pixels (default: 1_000_000).",
    )
    parser.add_argument(
        "--cluster-sigma",
        type=float,
        default=50.0 / 3.0,
        help="Standard deviation for the Gaussian clusters in pixels (default: 50/3).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=60.0,
        help="DBSCAN epsilon radius in pixels for scikit-learn clustering (default: 60).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=16,
        help="DBSCAN min_samples parameter for scikit-learn clustering (default: 16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random number generator (default: 42).",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data.bin"),
        help="Path to the output coordinate binary file (default: data.bin).",
    )
    parser.add_argument(
        "--truth-file",
        type=Path,
        default=Path("truth.bin"),
        help="Path to the output truth label binary file (default: truth.bin).",
    )
    return parser.parse_args()


def generate_uniform_points(rng: np.random.Generator, count: int, width: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.uint32)

    coords = rng.uniform(0, width, size=(count, 2))
    coords = np.rint(coords, casting="unsafe")  # round to nearest integer pixel
    coords = np.clip(coords, 0, width - 1)
    return coords.astype(np.uint32)


def generate_gaussian_clusters(
    rng: np.random.Generator,
    cluster_count: int,
    points_per_cluster: int,
    width: int,
    sigma: float,
) -> np.ndarray:
    if cluster_count <= 0 or points_per_cluster <= 0:
        return np.empty((0, 2), dtype=np.uint32)

    centers = rng.uniform(0, width, size=(cluster_count, 2))
    clusters = []

    for center in centers:
        samples = rng.normal(loc=center, scale=sigma, size=(points_per_cluster, 2))
        samples = np.rint(samples, casting="unsafe")
        samples = np.clip(samples, 0, width - 1)
        clusters.append(samples.astype(np.uint32))

    if not clusters:
        return np.empty((0, 2), dtype=np.uint32)

    return np.vstack(clusters)


def write_data_file(path: Path, coords_yx: np.ndarray) -> None:
    coords_yx.astype(np.uint32).tofile(path)


def write_truth_file(path: Path, labels: np.ndarray) -> None:
    labels.astype(np.int32).tofile(path)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    uniform_coords = generate_uniform_points(rng, args.uniform_count, args.area_width)
    cluster_coords = generate_gaussian_clusters(
        rng,
        cluster_count=args.cluster_count,
        points_per_cluster=args.points_per_cluster,
        width=args.area_width,
        sigma=args.cluster_sigma,
    )

    if uniform_coords.size == 0 and cluster_coords.size == 0:
        raise ValueError("No points generated; adjust the generator parameters.")

    # Concatenate and shuffle to avoid ordering artifacts.
    all_coords_yx = np.vstack([uniform_coords, cluster_coords])
    rng.shuffle(all_coords_yx, axis=0)

    # Prepare float coordinates for clustering (x, y order for scikit-learn).
    coords_xy_float = all_coords_yx[:, [1, 0]].astype(np.float64)

    dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(coords_xy_float)

    # Persist data in requested layout: (y, x) pairs as uint32.
    write_data_file(args.data_file, all_coords_yx)
    write_truth_file(args.truth_file, labels)

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_labels = unique_labels[unique_labels != -1]
    noise_count = counts[unique_labels == -1].sum() if -1 in unique_labels else 0

    print(f"Generated {all_coords_yx.shape[0]} total points.")
    print(f"Uniform points: {uniform_coords.shape[0]}, clustered points: {cluster_coords.shape[0]}.")
    print(f"DBSCAN discovered {cluster_labels.size} clusters and {noise_count} noise points.")
    print(f"Data written to {args.data_file.resolve()} and labels to {args.truth_file.resolve()}.")


if __name__ == "__main__":
    main()
