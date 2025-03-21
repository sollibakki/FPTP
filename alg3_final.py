import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

# ---------- ALGORITHM 3: K-MEANS CLUSTERING ---------
# ---- CLUSTERING NATIONS BASED ON OVERALL STRENGTH ----

warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows with MKL")

# Load the dataset
data = pd.read_csv("FPTP_statistics.csv")

# Define features for clustering
features = ["score", "income", "development", "buildings", "manpower_balance",
            "stability", "adm_tech", "dip_tech", "mil_tech"]
data_for_clustering = data[features]

# K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=1)
data["cluster"] = kmeans.fit_predict(data_for_clustering)

# Evaluate clustering performance
inertia = kmeans.inertia_
silhouette_avg = silhouette_score(data_for_clustering, data["cluster"])
print(f"Inertia: {inertia:.2f}")
print(f"Silhouette Score: {silhouette_avg:.2f}", end="\n\n")

# Visualize the clusters using PCA
pca_2d = PCA(n_components=2, random_state=1)
reduced_data_2d = pca_2d.fit_transform(data_for_clustering)
data["pca1"] = reduced_data_2d[:, 0]
data["pca2"] = reduced_data_2d[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 7))
colors = ["blue", "green", "purple"]
for cluster in range(3):
    cluster_data = data[data["cluster"] == cluster]
    plt.scatter(cluster_data["pca1"], cluster_data["pca2"], label=f"Cluster {cluster}", c=colors[cluster])

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Nation Clusters Based on General Strength")
plt.legend()
plt.grid()
plt.show()

# Summarize the characteristics of each cluster
cluster_summary = data.groupby("cluster")[features].mean()

print("Summary of clusters: \n", cluster_summary.to_string())

# Visualize cluster composition as a bar chart
cluster_counts = data["cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
cluster_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.xlabel("Cluster")
plt.ylabel("Number of Nations")
plt.title("Cluster Composition: Number of Nations per Cluster")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Highlighting potential outliers in cluster 0
cluster_0_data = data[data["cluster"] == 0]
cluster_0_outliers = cluster_0_data[
    (cluster_0_data["score"] < cluster_0_data["score"].quantile(0.25)) |
    (cluster_0_data["score"] > cluster_0_data["score"].quantile(0.75))]

# Compare the outliers in Cluster 0 to the average characteristics of Cluster 0
outliers_comparison = cluster_0_outliers[features].mean() - cluster_0_data[features].mean()

# Create a DataFrame for better readability
outliers_comparison_df = pd.DataFrame({
    "Feature": features,
    "Difference (Outliers vs. Cluster 0 Average)": outliers_comparison.values
}).sort_values(by="Difference (Outliers vs. Cluster 0 Average)", ascending=False)

# Display the comparison
print("\nOutlier Feature Comparison in Cluster 0:")
print(outliers_comparison_df)

# Data in 3 components for 3D visualization
pca_3d = PCA(n_components=3, random_state=1)
data_3d = pca_3d.fit_transform(data_for_clustering)

# Add the 3D PCA components to the DataFrame
data["pca3_1"] = data_3d[:, 0]
data["pca3_2"] = data_3d[:, 1]
data["pca3_3"] = data_3d[:, 2]

# Create a 3D visualization of the clusters
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot each cluster in 3D space with a customizable view
for cluster in range(3):
    cluster_data = data[data["cluster"] == cluster]
    ax.scatter(cluster_data["pca3_1"], cluster_data["pca3_2"], cluster_data["pca3_3"],
               c=colors[cluster], label=f"Cluster {cluster}", s=100)

ax.view_init(30, 60)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
ax.set_title("3D Visualization of Nation Clusters")
ax.legend()
plt.show()
