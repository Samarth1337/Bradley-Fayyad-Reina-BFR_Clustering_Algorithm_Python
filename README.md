# Bradley-Fayyad-Reina-BFR_Clustering_Algorithm_Python

## Overview
In this project, I have implemented the Bradley-Fayyad-Reina (BFR) algorithm for clustering synthetic datasets. The goal of this assignment is to gain familiarity with the clustering process, different distance measurements, and the implementation of the BFR algorithm. The synthetic datasets are generated with random centroids and standard deviations to simulate normally distributed clusters, along with outliers for algorithm evaluation.

## Problem Statement
The problem revolves around implementing the BFR algorithm to cluster synthetic data contained. The algorithm involves maintaining three sets of points: Discard set (DS), Compression set (CS), and Retained set (RS). The challenge lies in efficiently clustering the data while considering outliers and dynamically adjusting cluster memberships.

## Dataset
The synthetic dataset contains data points with features and cluster assignments. The clusters are generated with random centroids and standard deviations to simulate normal distribution. Outliers are also included in the dataset to evaluate the algorithm's performance.

## Solution Approach
The BFR algorithm is implemented in Python, leveraging libraries like NumPy and scikit-learn. The algorithm proceeds through several steps, including initial clustering with K-Means, handling outliers, dynamically updating cluster memberships, and merging clusters based on distance metrics.

## Key Features

Dynamic Clustering: The algorithm dynamically adjusts cluster memberships based on Mahalanobis distance, allowing for flexible clustering in varying data distributions.

Outlier Handling: Outliers are identified and processed separately, ensuring robustness in the presence of noisy data.

Efficient Memory Usage: By maintaining three sets of points and dynamically updating clusters, the algorithm optimizes memory usage while handling large datasets.

## Technologies Used

Python: The implementation is done in Python for its ease of use and availability of libraries like NumPy and scikit-learn.

NumPy: Used for efficient numerical computations and array manipulations.

scikit-learn: Utilized for K-Means clustering and distance calculations.

## Evaluation
The performance of the implemented algorithm can be evaluated based on various metrics such as clustering accuracy, runtime efficiency, and scalability to larger datasets. Experimentation with different synthetic datasets can provide insights into the algorithm's robustness and effectiveness in different scenarios.

## Conclusion
The implementation of the Bradley-Fayyad-Reina (BFR) algorithm provides a versatile solution for clustering synthetic datasets, demonstrating efficient memory usage and robust clustering performance. By dynamically adjusting cluster memberships and handling outliers, the algorithm offers a practical approach to clustering in real-world scenarios.
