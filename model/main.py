import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_image_paths(dir_path, category):
    category_path = os.path.join(dir_path, category)
    all_images = [
        f
        for f in os.listdir(category_path)
        if os.path.isfile(os.path.join(category_path, f))
    ]
    selected_image_paths = [os.path.join(category_path, img) for img in all_images]

    return selected_image_paths


def load_images(image_paths):
    images = []
    for path in image_paths:
        with Image.open(path) as img:
            img = img.convert("L").resize((64, 64))
            images.append(np.array(img).flatten())
    return np.array(images)


def calculate_and_save_covariance(data, file_name="covariance_matrix.txt"):
    mean_centered_data = data - np.mean(data, axis=0)
    covariance_matrix = np.cov(mean_centered_data, rowvar=False)
    np.savetxt(file_name, covariance_matrix)
    return covariance_matrix


def calculate_eigenvalues_eigenvectors(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors


def top_50_eigenvalues(eigenvalues):
    plt.figure(figsize=(10, 5))
    plt.plot(eigenvalues[:50], "o-")
    plt.title("Top 50 Eigenvalues")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.show()


def top_10_eigenfaces(eigenvectors, prefix="eigenface", n_eigenfaces=10):
    top_10_eigenvectors = eigenvectors[:n_eigenfaces]
    eigenfaces = top_10_eigenvectors.reshape(
        (10, 64, 64)
    )  # Reshape to 64x64 for each eigenface

    fig, axes = plt.subplots(
        2, 5, figsize=(15, 6)
    )  # 2 rows, 5 columns of subplots for the top 10 eigenfaces
    fig.suptitle("Top 10 Eigenfaces", fontsize=16)

    for i, eigenface in enumerate(eigenfaces):
        ax = axes[i // 5, i % 5]
        ax.imshow(eigenface, cmap="gray")
        ax.set_title(f"{prefix} {i+1}")
        ax.axis("off")  # Hide axis

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{prefix}_combined.png")
    plt.show()
    plt.close()


def project_to_new_space(data, eigenvectors):
    return np.dot(data - np.mean(data, axis=0), eigenvectors.T)


def process_dataset(dir_path, categories, n_components=10, calculate_covariance=True):
    all_images = []
    selected_files_info = {}

    # load images
    for category in categories:
        image_paths = get_image_paths(dir_path, category)  
        selected_files_info[category] = [os.path.basename(path) for path in image_paths]
        category_images = load_images(image_paths)
        all_images.append(category_images)

    all_images = np.vstack(all_images) 

    #  calc cov matrix
    if calculate_covariance:
        covariance_matrix = calculate_and_save_covariance(
            all_images, f"{dir_path}_covariance_matrix.txt"
        )
        # calc  eigenvalues & eigenvectors
        eigenvalues, eigenvectors = calculate_eigenvalues_eigenvectors(
            covariance_matrix
        )
    else:
        # Get eigenvalues and eigenvectors directly with PCA
        pca = PCA(n_components=n_components)
        pca.fit(all_images)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_

    # top 50 eingenvalues
    top_50_eigenvalues(eigenvalues)

    # top eingen faces
    top_10_eigenfaces(eigenvectors, prefix=f"{dir_path}_eigenface")

    # Trace and save images to a new vector space
    projected_data = project_to_new_space(all_images, eigenvectors)
    np.savetxt(f"{dir_path}_projected_data.txt", projected_data)

    for category, files in selected_files_info.items():
        print(f"Selected files for {category}: {files}")


# Example usage
dir_path = "."  # Update with your actual images path
categories = ["happy", "sad"]  # images dir name
categories_2 = ["mutlu","uzgun"]
process_dataset(
    dir_path, categories, 10, calculate_covariance=False)  # To use PCA, set it to false

process_dataset(dir_path, categories_2, 20, calculate_covariance=True)
