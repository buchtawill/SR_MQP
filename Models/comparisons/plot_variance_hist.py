#!/bin/mqp_venv/bin/python

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_tensor(file_path):
    return torch.load(file_path)

def compute_variances(tensor):
    variances = np.zeros(tensor.shape[0])
    for t in tqdm(range(len(tensor))):
        tens = tensor[t]
        variances[t] = tens.var().item()
    return variances

def plot_histogram(variances):
    plt.hist(variances, bins=60, color='red', edgecolor='black')
    plt.title('Histogram of Variances')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')

    plt.savefig('./images/var.png')
    
def plot_pca(tensors):
    # Reshape tensors from (num_samples, 3, 32, 32) to (num_samples, 3*32*32)
    num_samples = tensors.shape[0]
    flattened_tensors = tensors.view(num_samples, -1).numpy()

    # Perform PCA to reduce dimensions to 2
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(flattened_tensors)

    # Plot PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], s=1, alpha=0.5)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection of Tensor Set")
    plt.savefig("./images/pca.png")
    plt.close()
    
if __name__ == '__main__':
    tensor_path = '../data/data/low_res_tensors.pt'
    tensor = load_tensor(tensor_path)
    
    # print(tensor.shape) # ([702960, 3, 32, 32])
    
    num_samples = 5000
    random_indices = torch.randperm(tensor.shape[0])[:num_samples]
    random_subset = tensor[random_indices]
    
    variances = compute_variances(random_subset)
    plot_histogram(variances)
    
    # plot_pca(random_subset)
    