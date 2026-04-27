import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("FINAL/wine_cleaned.csv")

# -----------------------------
# PREP DATA
# -----------------------------
X = df.drop(columns=["quality"])

# scale features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# K-MEANS MODEL
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# -----------------------------
# 1. CLUSTER DISTRIBUTION
# -----------------------------
cluster_counts = df["cluster"].value_counts().sort_index()

plt.figure()
plt.bar(cluster_counts.index, cluster_counts.values)
plt.title("Wine Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.xticks(cluster_counts.index)
plt.show()

# -----------------------------
# 2. CLUSTER vs QUALITY
# -----------------------------
plt.figure()
plt.scatter(df["cluster"], df["quality"])
plt.title("Cluster vs Wine Quality")
plt.xlabel("Cluster")
plt.ylabel("Quality")
plt.show()

# -----------------------------
# 3. PCA VISUALIZATION (2D)
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["pc1"] = X_pca[:, 0]
df["pc2"] = X_pca[:, 1]

plt.figure()
plt.scatter(df["pc1"], df["pc2"], c=df["cluster"])
plt.title("K-Means Clusters (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -----------------------------
# 4. CLUSTER MEANS vs QUALITY
# -----------------------------
print("\nAverage quality per cluster:")
print(df.groupby("cluster")["quality"].mean())

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_csv("FINAL/wine_with_clusters.csv", index=False)

print("\nDone. File saved as wine_with_clusters.csv")