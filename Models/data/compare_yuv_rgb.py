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

def plot_rgb_yuv(rgb_tensor, yuv_tensor, idx):
    # Get the RGB and Y, U, V channel images for a given index in the batch
    rgb_image = rgb_tensor[idx].permute(1, 2, 0).numpy()  # Shape (H, W, 3)
    y_channel = yuv_tensor[idx, 0, :, :].numpy()  # Y channel (H, W)
    u_channel = yuv_tensor[idx, 1, :, :].numpy()  # U channel (H, W)
    v_channel = yuv_tensor[idx, 2, :, :].numpy()  # V channel (H, W)

    # Plot the RGB and YUV channels side by side
    fig, ax = plt.subplots(2, 2, figsize=(5, 5))
    
    # Display RGB image
    ax[0, 0].imshow(np.clip(rgb_image, 0, 1))  # Clip to ensure values are in range [0, 1]
    ax[0, 0].set_title(f'RGB Image {idx+1}')
    ax[0, 0].axis('off')
    
    # Display Y (luminance) channel as grayscale (top-right)
    ax[0, 1].imshow(y_channel, cmap='gray')
    ax[0, 1].set_title(f'Y Channel {idx+1}')
    ax[0, 1].axis('off')
    
    # Display U channel as grayscale (bottom-left)
    ax[1, 0].imshow(u_channel, cmap='gray')
    ax[1, 0].set_title(f'U Channel {idx+1}')
    ax[1, 0].axis('off')
    
    # Display V channel as grayscale (bottom-right)
    ax[1, 1].imshow(v_channel, cmap='gray')
    ax[1, 1].set_title(f'V Channel {idx+1}')
    ax[1, 1].axis('off')
    
    # Show the figure
    plt.tight_layout()
    
    # Show the figure
    plt.show()

def yuv_to_rgb_batch(yuv_batch):
    # yuv_batch: (N, 3, H, W)
    # Step 1: Permute to (N, H, W, 3) for matrix multiplication
    yuv_batch = yuv_batch.permute(0, 2, 3, 1)  # Shape (N, H, W, 3)
    
    # Step 2: Apply YUV to RGB conversion
    rgb_batch = torch.matmul(yuv_batch, yuv_to_rgb.T)  # Shape (N, H, W, 3)
    
    # Step 3: Permute back to (N, 3, H, W)
    return rgb_batch.permute(0, 3, 1, 2)

if __name__ == '__main__':
    rgb_tensor = torch.load("./rgb_5_images_64x64.pt", weights_only=True)
    yuv_tensor = torch.load("./yuv_5_images_64x64.pt",  weights_only=True)
    
    # print(rgb_tensor.shape)
    # print(yuv_tensor.shape)
    
    y_channel_only = yuv_tensor[:, 0, :, :]
    uv_channels = yuv_tensor[:, 1:, :, :]
    merged_yuv = torch.cat((y_channel_only.unsqueeze(1), uv_channels), dim=1)
    print(uv_channels.shape)
    print(y_channel_only.shape)
    print(merged_yuv.shape)
    
    back_to_rgb = yuv_to_rgb_batch(merged_yuv)
    # plt.imshow(back_to_rgb[1].permute(1, 2, 0).numpy())
    plt.imshow(rgb_tensor[1].permute(1, 2, 0).numpy())
    
    plt.show()
    exit()
    
    for i in range(len(rgb_tensor)-1):
        # print(rgb_tensor[i].shape) # 3 x 64 x 64
        # print(yuv_tensor[i].shape) # 3 x 64 x 64
        rgb = rgb_tensor#[i]
        # yuv = yuv_tensor#[i]
        plot_rgb_yuv(rgb, merged_yuv, i+1)