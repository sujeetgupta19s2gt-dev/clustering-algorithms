# Clustering Algorithms Practice

A hands-on repository covering core unsupervised machine learning algorithms implemented with Python, Scikit-Learn, and Seaborn.

---

## Notebook Details

### 1. KMeans.ipynb
* **Dataset**: Synthetic multi-cluster dataset.
* **Algorithm**: Standard **K-Means Clustering**.
* **Key Steps**:
  * Generated synthetic cluster points to study centroid-based partitioning.
  * Initialized and fitted `KMeans` to group data into distinct spherical clusters.
  * Visualized resulting cluster assignments and centroid locations using `seaborn.scatterplot`.

### 2. KMeansForIris.ipynb
* **Dataset**: Scikit-Learn's classic **Iris Dataset** (`load_iris()`, 150 samples across 4 features).
* **Algorithm**: **K-Means Clustering** with Feature Scaling.
* **Key Steps**:
  * Extracted feature matrix ($X \in \mathbb{R}^{150 \times 4}$) and target labels ($y$).
  * Visualized feature relationships prior to clustering (Sepal Length vs. Petal Length).
  * Applied **`StandardScaler()`** (`fit_transform`) to normalize feature scales before applying K-Means.

### 3. hierarchical_clustering.ipynb
* **Dataset**: Synthetic multi-dimensional feature data.
* **Algorithm**: **Agglomerative Hierarchical Clustering**.
* **Key Steps**:
  * Performed bottom-up hierarchical cluster merging.
  * Evaluated linkage criteria (e.g., Ward, Complete, Average) to determine inter-cluster distances.
  * Generated dendrogram visualizations to analyze cluster hierarchy without specifying K in advance.

### 4. DBSCAN.ipynb
* **Dataset**: Synthetic non-linear data (**`make_moons`** crescent dataset).
* **Algorithm**: **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise).
* **Key Steps**:
  * Handled non-linearly separable structures where centroid-based methods fail.
  * Configured density parameters `eps` (neighborhood radius) and `min_samples` (minimum density threshold).
  * Segmented intertwining crescent shapes and isolated noise/outlier points.

---

## Environment & Dependencies

* **Python 3.x**
* `scikit-learn`
* `pandas`
* `seaborn`
* `matplotlib`
