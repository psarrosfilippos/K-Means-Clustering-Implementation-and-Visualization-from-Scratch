#Psarros Filippos 
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b)**2))

def calculate_sse(Data, IDC, ClusterCenters):
    """Calculate the Sum of Squared Error (SSE)."""
    sse = 0
    for k in range(len(ClusterCenters)):
        # Select points that belong to each cluster
        cluster_points = Data[IDC == k]
        # Add squared distance from cluster center to SSE
        sse += np.sum((cluster_points - ClusterCenters[k])**2)
    return sse

def mykmeans(Data, K, max_iters=100, tol=1e-3):
    """
    Implementation of the K-means algorithm with:
    - Random initialization of centers from the data.
    - Termination when the movement of centers <= tol.
    """
    np.random.seed(42)  # For reproducibility
    N, M = Data.shape  # Number of points (N) and features (M)
    # Random selection of initial centers from data
    ClusterCenters = Data[np.random.choice(N, K, replace=False)]
    prev_centers = np.zeros_like(ClusterCenters)  # Initialize to store previous centers
    IDC = np.zeros(N, dtype=int)  # Initialize labels for each point
    sse_list = []  # List to store SSE for each iteration

    for iteration in range(max_iters):
        # Assign each point to the closest center
        for i in range(N):
            distances = [euclidean_distance(Data[i], center) for center in ClusterCenters]
            IDC[i] = np.argmin(distances)  # The cluster with the smallest distance

        prev_centers = ClusterCenters.copy()  # Store the previous centers

        # Update centers as the mean of points in each cluster
        for k in range(K):
            points_in_cluster = Data[IDC == k]
            if len(points_in_cluster) > 0:  # If there are points in the cluster
                ClusterCenters[k] = points_in_cluster.mean(axis=0)

        # Calculate SSE for the current iteration
        sse = calculate_sse(Data, IDC, ClusterCenters)
        sse_list.append(sse)  # Append SSE to the list
        print(f"Iteration {iteration + 1}: SSE = {sse:.5f}")

        # Check for termination if center movement <= tol
        center_shift = np.linalg.norm(ClusterCenters - prev_centers)
        if center_shift <= tol:
            print(f"Convergence reached at iteration {iteration + 1}")
            break

    # Plot all clusters after algorithm completion
    for iteration in range(len(sse_list)):
        plot_clusters(Data, IDC, ClusterCenters, iteration + 1)

    # Plot SSE graph at the end
    plot_sse(sse_list)
    return ClusterCenters, IDC

def plot_clusters(Data, IDC, ClusterCenters, iteration):
    """Plot the data points and centers for each iteration."""
    plt.figure()
    colors = ['red', 'green', 'blue']
    for k in range(len(ClusterCenters)):
        # Plot points in each cluster
        plt.scatter(Data[IDC == k, 0], Data[IDC == k, 1], c=colors[k], s=30, label=f'Cluster {k+1}')
        # Plot the center of the cluster
        plt.scatter(ClusterCenters[k, 0], ClusterCenters[k, 1], c='black', marker='+', s=200, label=f'Center {k+1}')
    plt.title(f'K-means: Iteration {iteration}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid()
    plt.show()

def plot_sse(sse_list):
    """Plot the Sum of Squared Error (SSE) for each iteration."""
    plt.figure()
    # Plot SSE graph
    plt.plot(range(1, len(sse_list) + 1), sse_list, marker='o', linestyle='-')
    plt.title('Sum of Squared Error (SSE) per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('SSE')
    plt.grid()
    plt.show()

def generate_data():
    """Generate data from three normal distributions."""
    np.random.seed(0)  # For reproducibility
    # Parameters of the three normal distributions
    mean_1, cov_1 = [4, 0], [[0.29, 0.4], [0.4, 0.4]]
    mean_2, cov_2 = [5, 7], [[0.29, 0.4], [0.4, 0.9]]
    mean_3, cov_3 = [7, 4], [[0.64, 0], [0, 0.64]]

    # Generate data from the distributions
    X1 = np.random.multivariate_normal(mean_1, cov_1, 50)
    X2 = np.random.multivariate_normal(mean_2, cov_2, 50)
    X3 = np.random.multivariate_normal(mean_3, cov_3, 50)

    # Merge the data
    Data = np.vstack((X1, X2, X3))
    return Data

if __name__ == "__main__":
    # Generate data
    Data = generate_data()
    K = 3  # Number of clusters

    # Run K-means
    ClusterCenters, IDC = mykmeans(Data, K)

    # Final display of the data and centers
    print("Final cluster centers:")
    print(ClusterCenters)
