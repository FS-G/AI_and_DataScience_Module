# Unsupervised Learning: Complete Lecture Guide

## Part 1: Foundations

### What is Unsupervised Learning?

Unsupervised learning is a branch of machine learning that finds hidden patterns in data without labeled examples. Unlike supervised learning where we have input-output pairs, unsupervised learning works with only input data, discovering underlying structures and relationships.

**Key Characteristics:**
- No target variable or ground truth labels
- Goal is to discover hidden patterns, structures, or relationships
- Often used for exploratory data analysis
- Results require interpretation and domain knowledge

### Types of Machine Learning

**Supervised Learning:**
- Has labeled training data (input-output pairs)
- Goal: Learn mapping function f(x) → y
- Examples: Classification, Regression
- Performance can be measured objectively

**Unsupervised Learning:**
- No labeled data, only input features
- Goal: Discover hidden patterns or structures
- Examples: Clustering, Dimensionality Reduction
- Performance evaluation is subjective

**Semi-supervised Learning:**
- Combination of labeled and unlabeled data
- Leverages large amounts of unlabeled data with small labeled dataset
- Examples: Self-training, Co-training

### Importance of Unsupervised Learning

**Data Exploration:**
- Understanding data distribution and structure
- Identifying outliers and anomalies
- Finding natural groupings in data

**Preprocessing:**
- Feature extraction and selection
- Noise reduction
- Data compression

**Discovery:**
- Market segmentation
- Gene sequencing analysis
- Social network analysis
- Recommendation systems

### The No Free Lunch Theorem

The No Free Lunch theorem states that there is no single algorithm that works best for all problems. This applies to unsupervised learning as well:

- K-Means works well for spherical clusters
- DBSCAN handles arbitrary shapes and outliers
- Hierarchical clustering reveals cluster hierarchy
- Choice depends on data characteristics and domain requirements

### Data Preprocessing

#### Scaling and Normalization - Mathematics

**Min-Max Normalization (Feature Scaling):**

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

Scales features to range [0, 1]

**Z-Score Standardization:**

$$X_{std} = \frac{X - \mu}{\sigma}$$

Where:
- μ = mean of the feature
- σ = standard deviation of the feature

Results in features with mean = 0 and std = 1

**Robust Scaling:**

$$X_{robust} = \frac{X - Q_{median}}{Q_{75} - Q_{25}}$$

Uses median and interquartile range, less sensitive to outliers

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Sample data
data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [100, 200]])

# Min-Max Normalization
min_max_scaler = MinMaxScaler()
data_minmax = min_max_scaler.fit_transform(data)
print("Min-Max Normalized:")
print(data_minmax)

# Z-Score Standardization
standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data)
print("\nStandardized:")
print(data_standard)

# Robust Scaling
robust_scaler = RobustScaler()
data_robust = robust_scaler.fit_transform(data)
print("\nRobust Scaled:")
print(data_robust)
```

### Distance Metrics - Mathematics

#### Euclidean Distance

$$d_{euclidean}(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

Most common distance metric, represents straight-line distance in n-dimensional space.

#### Manhattan Distance (L1 Distance)

$$d_{manhattan}(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

Sum of absolute differences, also called "city block" distance.

#### Cosine Distance

$$d_{cosine}(x, y) = 1 - \frac{x \cdot y}{||x|| \cdot ||y||} = 1 - \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2} \sqrt{\sum_{i=1}^{n} y_i^2}}$$

Measures angle between vectors, useful for high-dimensional sparse data.

#### Minkowski Distance (Generalized)

$$d_{minkowski}(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$$

Where p is the order:
- p = 1: Manhattan distance
- p = 2: Euclidean distance
- p = ∞: Chebyshev distance

```python
import numpy as np
from scipy.spatial.distance import euclidean, manhattan, cosine, minkowski

# Two sample points
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Calculate different distances
euclidean_dist = euclidean(x, y)
manhattan_dist = manhattan(x, y)
cosine_dist = cosine(x, y)
minkowski_dist = minkowski(x, y, p=3)

print(f"Euclidean Distance: {euclidean_dist:.3f}")
print(f"Manhattan Distance: {manhattan_dist:.3f}")
print(f"Cosine Distance: {cosine_dist:.3f}")
print(f"Minkowski Distance (p=3): {minkowski_dist:.3f}")

# Manual calculation for verification
euclidean_manual = np.sqrt(np.sum((x - y)**2))
manhattan_manual = np.sum(np.abs(x - y))
print(f"\nManual Euclidean: {euclidean_manual:.3f}")
print(f"Manual Manhattan: {manhattan_manual:.3f}")
```

### Curse of Dimensionality

As the number of dimensions increases, several problems arise:

**Distance Concentration:**
- All points become approximately equidistant
- Nearest and farthest neighbors converge to similar distances
- Makes distance-based algorithms less effective

**Sparsity:**
- Data points become sparse in high-dimensional space
- Volume of high-dimensional space increases exponentially
- Most of the volume is concentrated in the "corners"

**Computational Complexity:**
- Algorithms become computationally expensive
- Memory requirements increase dramatically

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate curse of dimensionality
dimensions = [2, 5, 10, 20, 50, 100]
n_points = 1000

for dim in dimensions:
    # Generate random points
    points = np.random.randn(n_points, dim)
    
    # Calculate distances from origin
    distances = np.sqrt(np.sum(points**2, axis=1))
    
    print(f"Dimension {dim}:")
    print(f"  Mean distance: {distances.mean():.3f}")
    print(f"  Std distance: {distances.std():.3f}")
    print(f"  Ratio (std/mean): {distances.std()/distances.mean():.3f}")
    print()
```

## Part 2: Core Tasks & Algorithms

### I. Clustering

Clustering is the task of grouping similar data points together while keeping dissimilar points in different groups.

#### K-Means Clustering - Step-by-Step Example

K-Means is an iterative algorithm that groups data points into k clusters. Let's work through a simple example with 6 data points and k=2 clusters.

**Sample Dataset:**
| Point | X | Y |
|-------|---|---|
| A | 1 | 2 |
| B | 2 | 1 |
| C | 3 | 3 |
| D | 8 | 8 |
| E | 9 | 7 |
| F | 8 | 9 |

**Step 1: Initialize Centroids**
Let's randomly pick two points as initial centroids:
- Centroid 1 (C1): Point A = (1, 2)
- Centroid 2 (C2): Point D = (8, 8)

**Step 2: Calculate Distances and Assign Clusters**

For each point, calculate distance to both centroids and assign to nearest:

**Point A (1, 2):**
- Distance to C1: √[(1-1)² + (2-2)²] = 0
- Distance to C2: √[(1-8)² + (2-8)²] = √[49 + 36] = 9.22
- **Assigned to Cluster 1**

**Point B (2, 1):**
- Distance to C1: √[(2-1)² + (1-2)²] = √[1 + 1] = 1.41
- Distance to C2: √[(2-8)² + (1-8)²] = √[36 + 49] = 9.22
- **Assigned to Cluster 1**

**Point C (3, 3):**
- Distance to C1: √[(3-1)² + (3-2)²] = √[4 + 1] = 2.24
- Distance to C2: √[(3-8)² + (3-8)²] = √[25 + 25] = 7.07
- **Assigned to Cluster 1**

**Point D (8, 8):**
- Distance to C1: √[(8-1)² + (8-2)²] = √[49 + 36] = 9.22
- Distance to C2: √[(8-8)² + (8-8)²] = 0
- **Assigned to Cluster 2**

**Point E (9, 7):**
- Distance to C1: √[(9-1)² + (7-2)²] = √[64 + 25] = 9.43
- Distance to C2: √[(9-8)² + (7-8)²] = √[1 + 1] = 1.41
- **Assigned to Cluster 2**

**Point F (8, 9):**
- Distance to C1: √[(8-1)² + (9-2)²] = √[49 + 49] = 9.90
- Distance to C2: √[(8-8)² + (9-8)²] = √[0 + 1] = 1
- **Assigned to Cluster 2**

**After Step 2:**
- Cluster 1: {A, B, C}
- Cluster 2: {D, E, F}

**Step 3: Update Centroids**

Calculate new centroids as the mean of assigned points:

**New Centroid 1:**
- X = (1 + 2 + 3) / 3 = 2
- Y = (2 + 1 + 3) / 3 = 2
- New C1 = (2, 2)

**New Centroid 2:**
- X = (8 + 9 + 8) / 3 = 8.33
- Y = (8 + 7 + 9) / 3 = 8
- New C2 = (8.33, 8)

**Step 4: Check Convergence**
Compare old and new centroids:
- C1 moved from (1, 2) to (2, 2) - distance = 1
- C2 moved from (8, 8) to (8.33, 8) - distance = 0.33

If these movements are above our tolerance threshold, we repeat steps 2-3 with the new centroids.

**From Scratch Implementation:**

```python
import numpy as np

class KMeans:
    def __init__(self, k=2, max_iters=100, tol=1e-4, distance_metric='euclidean'):
        """
        Initialize the KMeans clustering algorithm.
        Parameters:
        k -- Number of clusters
        max_iters -- Maximum number of iterations
        tol -- Tolerance for convergence
        distance_metric -- Distance metric to use ('euclidean', 'manhattan', 'minkowski')
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.distance_metric = distance_metric
        self.centroids = None
        self.labels = None

    def _initialize_centroids(self, X):
        """Randomly initialize k centroids from the data points."""
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        centroids = X[indices]
        print(f"Initial Centroids:\n{centroids}\n")
        return centroids

    def _compute_distances(self, X):
        """Compute distances between each data point and each centroid."""
        distances = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            if self.distance_metric == 'euclidean':
                distances[:, i] = np.sqrt(np.sum((X - self.centroids[i]) ** 2, axis=1))
            elif self.distance_metric == 'manhattan':
                distances[:, i] = np.sum(np.abs(X - self.centroids[i]), axis=1)
            elif self.distance_metric == 'minkowski':
                distances[:, i] = np.sum(np.abs(X - self.centroids[i]) ** 3, axis=1) ** (1/3)
            else:
                raise ValueError("Unsupported distance metric")
        print(f"Distances:\n{distances}\n")
        return distances

    def _update_centroids(self, X, labels):
        """Recalculate the centroids based on the current cluster assignments."""
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])
        print(f"New Centroids:\n{new_centroids}\n")
        return new_centroids

    def fit(self, X):
        """Perform K-means clustering."""
        # step 1: randomly initialize the centroids
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            print(f"Iteration {iteration + 1}:")
            
            # Step 2: Assign clusters
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)
            print(f"Labels:\n{self.labels}\n")
            
            # Step 3: Calculate new centroids
            new_centroids = self._update_centroids(X, self.labels)
            
            # Check for convergence
            centroid_shifts = np.linalg.norm(new_centroids - self.centroids, axis=1)
            print(f"Centroid Shifts:\n{centroid_shifts}\n")
            if np.all(centroid_shifts < self.tol):
                print("Convergence reached.")
                break
                
            self.centroids = new_centroids

    def predict(self, X):
        """Predict the cluster for each data point."""
        distances = self._compute_distances(X)
        predictions = np.argmin(distances, axis=1)
        print(f"Predictions:\n{predictions}\n")
        return predictions

# Sample data
X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]])

# Number of clusters
k = 2

# Initialize and fit K-Means
kmeans = KMeans(k=k, distance_metric='manhattan')  # You can change to 'euclidean' or 'minkowski'
kmeans.fit(X)

# Print results
print("Final Centroids:\n", kmeans.centroids)
print("Final Labels:\n", kmeans.labels)

# Predict cluster assignments for new data
X_new = np.array([[2.0, 3.0], [6.0, 7.0]])
predictions = kmeans.predict(X_new)
print("Predictions for new data:\n", predictions)
```

**Choosing Optimal K:**

**Elbow Method:**
Plot WCSS vs number of clusters and look for the "elbow" point.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 2)

# Calculate WCSS for different k values
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
```

**Silhouette Analysis:**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- a(i) = average intra-cluster distance
- b(i) = average nearest-cluster distance
- Range: [-1, 1], higher is better

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

# Calculate silhouette scores for different k values
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"k={k}, Silhouette Score: {silhouette_avg:.3f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'ro-')
plt.title('Silhouette Analysis for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.grid(True)
plt.show()
```

#### Hierarchical Clustering

**Agglomerative (Bottom-up):**
- Start with each point as individual cluster
- Iteratively merge closest clusters
- Creates hierarchical tree structure

**Divisive (Top-down):**
- Start with all points in one cluster
- Recursively split clusters
- Less commonly used

**Linkage Criteria:**
- **Single Linkage:** Minimum distance between clusters
- **Complete Linkage:** Maximum distance between clusters
- **Average Linkage:** Average distance between all pairs
- **Ward Linkage:** Minimizes within-cluster variance

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.array([[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]])

# Perform hierarchical clustering
linkage_matrix = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# Apply clustering with specific number of clusters
agg_clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
cluster_labels = agg_clustering.fit_predict(X)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Agglomerative Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("Cluster labels:", cluster_labels)
```

#### DBSCAN (Density-Based Spatial Clustering)

DBSCAN groups together points in high-density areas and marks points in low-density areas as outliers.

**Key Concepts:**
- **Core Point:** Has at least MinPts points within distance ε
- **Border Point:** Within ε distance of a core point but not core itself
- **Noise Point:** Neither core nor border point (outlier)

**Parameters:**
- **ε (epsilon):** Maximum distance between two samples for them to be considered neighbors
- **MinPts:** Minimum number of points required to form a dense region

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data with noise
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, 
                  random_state=42, cluster_std=0.60)

# Add noise points
noise = np.random.uniform(-6, 6, (50, 2))
X = np.vstack([X, noise])

# Apply DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
cluster_labels = dbscan.fit_predict(X)

# Count clusters and noise points
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")

# Plot results
plt.figure(figsize=(10, 8))
unique_labels = set(cluster_labels)
colors = ['red' if label == -1 else plt.cm.Spectral(label / len(unique_labels)) 
          for label in cluster_labels]

for label, color in zip(cluster_labels, colors):
    if label == -1:
        # Black used for noise
        plt.scatter(X[cluster_labels == label, 0], X[cluster_labels == label, 1], 
                   c='black', marker='x', s=50, label='Noise' if label == -1 else f'Cluster {label}')
    else:
        plt.scatter(X[cluster_labels == label, 0], X[cluster_labels == label, 1], 
                   c=[color], s=50, label=f'Cluster {label}')

plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

### II. Dimensionality Reduction

High-dimensional data can be challenging to work with due to computational complexity, visualization difficulties, and the curse of dimensionality. Dimensionality reduction techniques help by projecting data to lower dimensions while preserving important information.

**Why Dimensionality Reduction?**
- **Visualization:** Reduce to 2D/3D for plotting
- **Computational Efficiency:** Fewer features = faster algorithms
- **Storage:** Reduce memory requirements
- **Noise Removal:** Focus on most important features
- **Feature Engineering:** Create better representations

#### Principal Component Analysis (PCA)

PCA finds the directions (principal components) of maximum variance in the data.

**Key Concepts:**
- **Principal Components:** Orthogonal directions of maximum variance
- **Eigenvalues:** Amount of variance explained by each component
- **Explained Variance Ratio:** Proportion of total variance explained

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load sample data
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Print explained variance ratio
print("Explained Variance Ratio:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.3f}")

print(f"Cumulative Explained Variance: {pca.explained_variance_ratio_.cumsum()}")

# Visualize results
plt.figure(figsize=(15, 5))

# Plot original features
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Original Features')

# Plot first two principal components
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA - First Two Components')

# Plot explained variance
plt.subplot(1, 3, 3)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Component')

plt.tight_layout()
plt.show()

# Reduce to 2 dimensions
pca_2d = PCA(n_components=2)
X_reduced = pca_2d.fit_transform(X)
print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Variance retained: {pca_2d.explained_variance_ratio_.sum():.3f}")
```

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is excellent for visualizing high-dimensional data by preserving local structure.

**Key Parameters:**
- **Perplexity:** Balance between local and global aspects (typically 5-50)
- **Learning Rate:** Step size for optimization (typically 10-1000)
- **Iterations:** Number of optimization iterations

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load sample data (handwritten digits)
digits = load_digits()
X = digits.data
y = digits.target

# Apply t-SNE with different perplexity values
perplexities = [5, 30, 50]

plt.figure(figsize=(15, 5))

for i, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    plt.subplot(1, 3, i+1)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
    plt.title(f't-SNE (perplexity={perplexity})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter)

plt.tight_layout()
plt.show()
```

#### UMAP (Uniform Manifold Approximation and Projection)

UMAP is faster than t-SNE and preserves both local and global structure better.

```python
# Note: Install umap-learn first: pip install umap-learn
try:
    import umap.umap_ as umap
    
    # Apply UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X)
    
    plt.figure(figsize=(12, 5))
    
    # Compare t-SNE and UMAP
    plt.subplot(1, 2, 1)
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X[:1000])  # Use subset for speed
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y[:1000], cmap='tab10')
    plt.title('t-SNE')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_umap[:1000, 0], X_umap[:1000, 1], c=y[:1000], cmap='tab10')
    plt.title('UMAP')
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("UMAP not installed. Install with: pip install umap-learn")
```



### III. Association Rule Mining

Association rule mining finds relationships between different items in transactional data. Most commonly used in market basket analysis.

**Key Concepts:**

**Support:** Frequency of itemset in dataset
$$Support(A) = \frac{\text{Transactions containing A}}{\text{Total transactions}}$$

**Confidence:** Likelihood of B given A
$$Confidence(A \Rightarrow B) = \frac{Support(A \cup B)}{Support(A)}$$

**Lift:** Measures how much more likely B is to occur when A occurs, compared to B occurring randomly

$Lift(A \Rightarrow B) = \frac{Confidence(A \Rightarrow B)}{Support(B)} = \frac{P(B|A)}{P(B)}$

**Lift Interpretation:**
- **Lift = 1:** A and B are independent (no relationship)
- **Lift > 1:** A and B occur together more often than expected by chance (positive correlation)
- **Lift < 1:** A and B occur together less often than expected by chance (negative correlation)

**Example:** If 30% of customers buy bread, but 60% of customers who buy butter also buy bread:
- Support(bread) = 0.30
- Confidence(butter → bread) = 0.60
- Lift(butter → bread) = 0.60 / 0.30 = 2.0

This means customers who buy butter are **2 times more likely** to buy bread than a random customer.

**Use Cases:**
- **Market Basket Analysis:** "People who buy bread also buy butter"
- **Recommendation Systems:** Suggest products based on purchase history
- **Web Usage Mining:** Understand user navigation patterns
- **Bioinformatics:** Find gene associations
- **Text Mining:** Discover word co-occurrence patterns

**Market Basket Analysis Example:**

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Sample transaction data
transactions = [
    ['milk', 'eggs', 'bread', 'cheese'],
    ['eggs', 'bread'],
    ['milk', 'bread'],
    ['eggs', 'bread', 'butter'],
    ['milk', 'eggs', 'bread', 'cheese'],
    ['milk', 'eggs', 'bread', 'butter'],
    ['eggs', 'bread'],
    ['milk', 'bread', 'butter'],
    ['milk', 'eggs', 'cheese'],
    ['bread', 'butter']
]

print("Sample Transactions:")
for i, transaction in enumerate(transactions[:5], 1):
    print(f"Transaction {i}: {transaction}")

# Convert to binary format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("\nBinary Transaction Matrix:")
print(df.head())

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print(f"\nFrequent Itemsets (min_support=0.3):")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(f"\nAssociation Rules (min_confidence=0.5):")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Interpret results
print("\nRule Interpretation:")
for idx, rule in rules.iterrows():
    antecedent = ', '.join(list(rule['antecedents']))
    consequent = ', '.join(list(rule['consequents']))
    print(f"Rule: {antecedent} → {consequent}")
    print(f"  Support: {rule['support']:.3f} (appears in {rule['support']*100:.1f}% of transactions)")
    print(f"  Confidence: {rule['confidence']:.3f} ({rule['confidence']*100:.1f}% chance of buying {consequent} if buying {antecedent})")
    print(f"  Lift: {rule['lift']:.3f} ({rule['lift']:.1f}x more likely than random)")
    print()
```

**Algorithm Implementations:**

```python
# Apriori Algorithm (Conceptual Implementation)
def generate_candidates(prev_frequent, k):
    """Generate candidate itemsets of size k from frequent itemsets of size k-1"""
    candidates = []
    n = len(prev_frequent)
    
    for i in range(n):
        for j in range(i+1, n):
            # Join step: merge itemsets that differ by one item
            union = prev_frequent[i].union(prev_frequent[j])
            if len(union) == k:
                candidates.append(union)
    
    return candidates

def calculate_support(transactions, itemsets):
    """Calculate support for given itemsets"""
    itemset_counts = {}
    total_transactions = len(transactions)
    