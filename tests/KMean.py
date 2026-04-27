import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from skimage.io import imread
import warnings; warnings.simplefilter('ignore')  
import os
from mpl_toolkits.mplot3d import Axes3D


image_path = os.path.join('tests', 'images', 'bird1.jpg')

print("Current Directory:", os.getcwd())
print("File Exists:", os.path.exists(image_path))

img = imread(image_path)
img = img / 255.0


original_shape = img.shape
pixels = img.reshape(-1, 3)

print("Original Shape:", original_shape)
print("Pixels Shape:", pixels.shape)


k = 3  
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
kmeans.fit(pixels)

compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
compressed_img = compressed_pixels.reshape(original_shape)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Compressed Image (k={k})")
plt.imshow(compressed_img)
plt.axis('off')

plt.show()


def plot_pixels(pixels, title, colors=None, N=10000):
    if colors is None:
        colors = pixels
    
    # random sample
    rng = np.random.RandomState(0)
    i = rng.permutation(pixels.shape[0])[:N]
    
    sample_pixels = pixels[i]
    sample_colors = colors[i]
    
    R, G, B = sample_pixels.T
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    ax[0].scatter(R, G, color=sample_colors, marker='.', s=1)
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))
    
    ax[1].scatter(R, B, color=sample_colors, marker='.', s=1)
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
    
    fig.suptitle(title, size=20)
    plt.show()


plot_pixels(pixels, title="Original Color Space")


kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
kmeans.fit(pixels)

new_colors = kmeans.cluster_centers_[kmeans.predict(pixels)]

plot_pixels(pixels, colors=new_colors,
            title="Reduced Color Space (3 colors)")

def plot_pixels_3d(pixels, title, colors=None, N=5000):
    if colors is None:
        colors = pixels
    
    rng = np.random.RandomState(42)
    i = rng.permutation(pixels.shape[0])[:N]
    
    sample_pixels = pixels[i]
    sample_colors = colors[i]
    
    R, G, B = sample_pixels.T

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(R, G, B, c=sample_colors, marker='.', s=20)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title(title)

    plt.show()

plot_pixels_3d(pixels, title="3D Color Space (Original)")

kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)

new_colors = kmeans.cluster_centers_[kmeans.predict(pixels)]

plot_pixels_3d(pixels, colors=new_colors,
               title="3D Color Space (3 Clusters)")
