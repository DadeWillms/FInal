import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('cleaned_file.csv')

features = [
    'StudyHours', 'SleepHours', 'AttendancePercent', 'AssignmentScore', 
    'ScreenTimeHours', 'ExerciseHours', 'SocialMediaHours', 
    'ExamScore', 'ProjectsCompleted', 'PartTimeJobHours'
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

k = 3

kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis')
plt.show()

cluster_summary = df.groupby('Cluster')[features].mean()
print(cluster_summary)

centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
print(centers)

df.to_csv('clustered_output.csv', index=False)