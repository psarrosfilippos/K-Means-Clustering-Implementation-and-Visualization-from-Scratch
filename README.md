
# K-Means Clustering: Implementation and Visualization from Scratch

This project implements the K-Means clustering algorithm from scratch in Python, including manual calculation of Euclidean distance, cluster assignment, centroid updates, and Sum of Squared Errors (SSE) tracking per iteration.
The goal is to demonstrate the inner workings of K-Means on synthetic 2D data generated from three distinct Gaussian distributions, and visualize both the clustering process and convergence behavior.
## Features

**Manual K-Means Implementation**

Full implementation of the K-Means algorithm from scratch using NumPy, without relying on high-level libraries like scikit-learn.

**Custom Euclidean Distance Function**

Distance between points is calculated manually to show the core mathematical operation of clustering.

**Dynamic Cluster Center Updates**

Cluster centers are updated in each iteration as the mean of the points assigned to each cluster.

**Sum of Squared Errors (SSE) Tracking**

SSE is computed after every iteration to evaluate convergence and optimize the clustering process.

**Termination Based on Centroid Movement**

The algorithm stops early if the movement of centroids between iterations is below a given threshold (tol).

**Cluster Visualization per Iteration**

2D scatter plots are generated in every iteration showing point distribution, cluster assignments, and centroids.

**SSE Visualization**

A separate line plot shows the decrease of SSE over iterations, indicating the convergence of the algorithm.

**Synthetic Dataset Generation**

Generates synthetic data from three different multivariate normal distributions with custom means and covariances.

## Technologies Used

- **Python 3.10+** – Main programming language used for implementing the algorithm.

- **NumPy** – Used for vectorized operations, matrix manipulations, and numerical calculations.

- **Matplotlib** – Used to visualize clustering results and SSE progression across iterations.

- **Replit** – Online IDE used for development, execution, and visualization of the project.

## How to use

**1. Clone or Download the Repository**

Clone or download the main.py script to your local machine, or open it directly in Replit.

**2. Install Dependencies**

If running locally, install the required libraries using pip:

    pip install numpy matplotlib

**3. Run the Script**

Execute the script with:

    python main.py

The script will:

- Generate synthetic data from 3 normal distributions.

- Run the K-Means clustering algorithm.

- Display cluster assignments over multiple iterations.

- Visualize the Sum of Squared Error (SSE) convergence.

**4. Customize (Optional)**

You can experiment by:

Changing the K variable to use a different number of clusters.

Modifying the mean and cov values inside the generate_data() function to create new data distributions.

Alternatively, you can run the project directly online using Replit without any local setup.

## Example Output

After running the script, the following will be displayed:

**Cluster Plots for each iteration of the K-Means algorithm, showing:**

- Data points colored by assigned cluster

- Cluster centers marked with a black "+" symbol

- Iteration number in the title

**SSE Plot (Sum of Squared Error) across iterations:**

- A line graph showing how the SSE decreases as the algorithm converges

- Helps visualize the point of convergence

**Console Output:**

- For each iteration:


        Iteration 1: SSE = 123.45678
        Iteration 2: SSE = 95.43210
        ...
        Convergence achieved at iteration X

- Final output:

        Final cluster centers:
        [[x1, y1],
        [x2, y2],
        [x3, y3]]


## Acknowledgements

This project was developed during the 4th year of my undergraduate studies in Informatics and Telecommunications as part of my academic coursework.

I would like to thank my professors for their guidance and the constructive feedback throughout the development of this project. Special thanks also go to the academic community and open-source contributors whose tools and knowledge made this implementation possible.

Tools such as NumPy, Matplotlib, and Replit were essential in the development and visualization of this algorithm.

## Authors

Filippos Psarros

informatics and telecommunications Student

GitHub: psarrosfilippos
[README.md](https://github.com/user-attachments/files/21332341/README.md)
