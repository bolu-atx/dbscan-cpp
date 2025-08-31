#!/usr/bin/env python3

import numpy as np
import struct
import sys

def generate_test_data(n_samples=300, n_centers=4, output_file='test_data.bin'):
    """
    Generate test data with known clusters and save to binary file.
    Creates simple clusters manually without sklearn.
    """
    return generate_test_data_with_params(n_samples, n_centers, 0.8, 5, output_file)

def load_test_data(input_file='test_data.bin'):
    """
    Load test data from binary file.
    Returns: points (N, 2), labels
    """
    with open(input_file, 'rb') as f:
        # Read number of points
        n_points = struct.unpack('I', f.read(4))[0]

        points = []
        labels = []

        # Read points
        for _ in range(n_points):
            x, y = struct.unpack('dd', f.read(16))
            points.append([x, y])

        # Read labels
        for _ in range(n_points):
            label = struct.unpack('i', f.read(4))[0]
            labels.append(label)

    return np.array(points), np.array(labels)

def generate_multiple_test_cases():
    """Generate test cases with different data sizes and parameters"""
    test_cases = [
        # (n_samples, n_centers, eps, min_samples, output_file)
        (500, 3, 0.8, 5, 'test_data_500.bin'),
        (10000, 5, 0.8, 5, 'test_data_10k.bin'),
        (100000, 8, 0.8, 5, 'test_data_100k.bin'),  # Using 100k instead of 1M for practicality
        # Different eps values
        (1000, 4, 0.5, 5, 'test_data_eps_0_5.bin'),
        (1000, 4, 1.2, 5, 'test_data_eps_1_2.bin'),
        # Different min_samples values
        (1000, 4, 0.8, 3, 'test_data_minpts_3.bin'),
        (1000, 4, 0.8, 10, 'test_data_minpts_10.bin'),
    ]

    for n_samples, n_centers, eps, min_samples, output_file in test_cases:
        print(f"Generating {n_samples} points with {n_centers} centers (eps={eps}, min_samples={min_samples})...")
        generate_test_data_with_params(n_samples, n_centers, eps, min_samples, output_file)
        print(f"Saved to {output_file}\n")

def generate_test_data_with_params(n_samples=300, n_centers=4, eps=0.8, min_samples=5, output_file='test_data.bin'):
    """
    Generate test data with specific DBSCAN parameters
    """
    np.random.seed(42)

    # Generate clustered data manually
    X = []
    true_labels = []

    for center_idx in range(n_centers):
        # Create cluster centers
        center_x = np.random.uniform(-5, 5)
        center_y = np.random.uniform(-5, 5)

        # Generate points around each center
        cluster_size = n_samples // n_centers
        for _ in range(cluster_size):
            x = center_x + np.random.normal(0, 0.6)
            y = center_y + np.random.normal(0, 0.6)
            X.append([x, y])
            true_labels.append(center_idx)

    # Add some noise points
    noise_points = n_samples // 10
    for _ in range(noise_points):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        X.append([x, y])
        true_labels.append(-1)  # Noise

    X = np.array(X)
    true_labels = np.array(true_labels)

    # Simple DBSCAN-like clustering for ground truth
    labels = np.full(len(X), -1)
    cluster_id = 0

    for i in range(len(X)):
        if labels[i] != -1:
            continue

        # Find neighbors within eps
        neighbors = []
        for j in range(len(X)):
            if i == j:
                continue
            dist = np.sqrt((X[i][0] - X[j][0])**2 + (X[i][1] - X[j][1])**2)
            if dist <= eps:
                neighbors.append(j)

        if len(neighbors) >= min_samples:  # min_samples
            labels[i] = cluster_id
            # Expand cluster (simplified)
            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
            cluster_id += 1
        else:
            labels[i] = -1

    # Save data to binary file
    with open(output_file, 'wb') as f:
        # Write number of points
        f.write(struct.pack('I', len(X)))

        # Write points as doubles (x, y)
        for point in X:
            f.write(struct.pack('dd', point[0], point[1]))

        # Write computed labels for validation
        for label in labels:
            f.write(struct.pack('i', label))

    print(f"Generated {len(X)} points with {n_centers} clusters")
    print(f"Simple clustering found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")

    return X, labels

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--multiple':
        generate_multiple_test_cases()
    elif len(sys.argv) > 1:
        output_file = sys.argv[1]
        generate_test_data(output_file=output_file)
    else:
        generate_test_data()