import os
import time
import torch
#from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

# Define the RGB to YUV conversion matrix and its inverse (YUV to RGB)
rgb_to_yuv = torch.tensor([[ 0.299,  0.587,  0.114],
                           [-0.14713, -0.28886,  0.436],
                           [ 0.615,  -0.51499, -0.10001]])

yuv_to_rgb = torch.inverse(rgb_to_yuv)

def rgb_to_yuv_batch(rgb_batch):
    # rgb_batch: (N, 3, H, W)
    # Step 1: Permute to (N, H, W, 3) for matrix multiplication
    rgb_batch = rgb_batch.permute(0, 2, 3, 1)  # Shape (N, H, W, 3)
    
    # Step 2: Apply RGB to YUV conversion
    yuv_batch = torch.matmul(rgb_batch, rgb_to_yuv.T)  # Shape (N, H, W, 3)
    
    # Step 3: Permute back to (N, 3, H, W)
    return yuv_batch.permute(0, 3, 1, 2)

def yuv_to_rgb_batch(yuv_batch):
    # yuv_batch: (N, 3, H, W)
    # Step 1: Permute to (N, H, W, 3) for matrix multiplication
    yuv_batch = yuv_batch.permute(0, 2, 3, 1)  # Shape (N, H, W, 3)
    
    # Step 2: Apply YUV to RGB conversion
    rgb_batch = torch.matmul(yuv_batch, yuv_to_rgb.T)  # Shape (N, H, W, 3)
    
    # Step 3: Permute back to (N, 3, H, W)
    return rgb_batch.permute(0, 3, 1, 2)

# Function to plot RGB and Y (luminance) of YUV side by side
def plot_rgb_yuv(rgb_tensor, yuv_tensor, idx):
    # Get the RGB and Y channel images for a given index in the batch
    rgb_image = rgb_tensor[idx].permute(1, 2, 0).numpy()  # Shape (H, W, 3)
    y_channel = yuv_tensor[idx, 0, :, :].numpy()  # Y channel (H, W)
    
    # Plot the RGB and Y channel side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display RGB image
    ax[0].imshow(np.clip(rgb_image, 0, 1))  # Clip to ensure values are in range [0, 1]
    ax[0].set_title(f'RGB Image {idx+1}')
    ax[0].axis('off')
    
    # Display Y (luminance) channel as grayscale
    ax[1].imshow(y_channel, cmap='gray')
    ax[1].set_title(f'YUV Image (Y Channel) {idx+1}')
    ax[1].axis('off')
    
    plt.savefig(f"./yuv_and_rgb{idx}.png")
    plt.close()

if __name__ == '__main__':
    
    time_before = time.time()
    high_res = torch.load("./yuv_test/high_res_yuv_all.pt")
    low_res = torch.load("./yuv_test/low_res_yuv_all.pt")
    
    print(f"INFO [test_tensor_process.py] It took {(time.time() - time_before):.2f} seconds to load")
    print(f"INFO [test_tensor_process.py] High res tensor shape: {high_res.shape}")
    print(f"INFO [test_tensor_process.py] Low res tensor shape:  {low_res.shape}")
    
    # high_res_yuv_small_sample = rgb_to_yuv_batch(high_res[0:5])
    # low_res_yuv_small_sample = rgb_to_yuv_batch(low_res[0:5])
    
    high_res_slice = high_res[0:5000].clone()
    low_res_slice  = low_res[0:5000].clone()
    print(f"INFO [test_tensor_process.py] high_res_slice x5000 tensor shape:  {high_res_slice.shape}")
    print(f"INFO [test_tensor_process.py] low_res_slice x5000 tensor shape:  {low_res_slice.shape}")
    torch.save(high_res_slice, "./yuv_test/yuv_5k_images_64x64.pt")
    torch.save(low_res_slice, "./yuv_test/yuv_5k_images_32x32.pt")
    
    # high_res_yuv_all = rgb_to_yuv_batch(high_res)
    # low_res_yuv_all = rgb_to_yuv_batch(low_res)
    
    # torch.save(low_res_yuv_all, "./yuv_test/low_res_yuv_all.pt")
    
    print(f"INFO [test_tensor_process.py] This script took {(time.time() - time_before):.2f} seconds")
    