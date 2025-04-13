from FSRCNN import *
import sys
import time
import torch
# import torchinfo
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
# from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from low_hi_res_dataset import SR_image_dataset
from low_hi_res_dataset import SR_tensor_dataset
from torch.utils.tensorboard import SummaryWriter

# For interpolation
from scipy.ndimage import zoom

NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARN_RATE = 0.0005
COLOR_SPACE = 'rgb'

# Define the RGB to YUV conversion matrix and its inverse (YUV to RGB)
rgb_to_yuv = torch.tensor([[ 0.299,  0.587,  0.114],
                           [-0.14713, -0.28886,  0.436],
                           [ 0.615,  -0.51499, -0.10001]])

yuv_to_rgb = torch.inverse(rgb_to_yuv)

def yuv_to_rgb_batch(yuv_batch):
    # yuv_batch: (N, 3, H, W)
    # Step 1: Permute to (N, H, W, 3) for matrix multiplication
    yuv_batch = yuv_batch.permute(0, 2, 3, 1)  # Shape (N, H, W, 3)
    
    # Step 2: Apply YUV to RGB conversion
    rgb_batch = torch.matmul(yuv_batch, yuv_to_rgb.T)  # Shape (N, H, W, 3)
    
    # Step 3: Permute back to (N, 3, H, W)
    return rgb_batch.permute(0, 3, 1, 2)

def tensor_to_image(tensor:torch.tensor) -> Image:
    return transforms.ToPILImage()(tensor)

def plot_images(low_res, inference, truths, title:str=None):
    #plot low res, non-normalized, normalized, nearest neighbor, truth
    low_res = low_res.cpu()
    inference = inference.cpu()
    truths = truths.cpu()
    
    if(COLOR_SPACE == 'yuv'):
        low_res = yuv_to_rgb_batch(low_res)
        inference = yuv_to_rgb_batch(inference)
        truths = yuv_to_rgb_batch(truths)

    fig, axs = plt.subplots(3, 4, figsize=(12, 8))
    for i in range(3):
        low_res_img = tensor_to_image(normalize_tensor_image(low_res[i]))
        axs[i, 0].set_title('Low Res')
        axs[i, 0].imshow(low_res_img)
        axs[i, 0].axis('off')
        
        axs[i, 1].set_title('Bilinear')
        axs[i, 1].imshow(zoom(low_res_img, (2, 2, 1), order=1))
        axs[i, 1].axis('off')
        
        axs[i, 2].set_title('Modeled, Normalized')
        axs[i, 2].imshow(tensor_to_image(normalize_tensor_image(inference[i])))
        axs[i, 2].axis('off')
        
        axs[i, 3].set_title('Truth')
        axs[i, 3].imshow(tensor_to_image(normalize_tensor_image(truths[i])))
        axs[i, 3].axis('off')
    plt.tight_layout()
    
    if(title is not None):
        plt.savefig(title)
    else:
        plt.show()
    plt.close()


def model_dataloader_inference(model, dataloader, device, criterion, optimizer):
    """
    Run the forward pass of model on all samples in dataloader with criterion loss. If optimizer is set to None,
    this function will NOT perform gradient updates or optimizations.
    Args:
        model(nn.Module): The neural network model
        dataloader(torch.utils.data.DataLoader): PyTorch dataloader 
        criterion(): Loss criterion (e.g. MSE loss)
        optimizer(torch.optim): Optimizer for NN
    """
    running_loss = 0.0
    for batch in dataloader:
        low_res, hi_res_truth = batch
        
        # print(f"INFO [model_dataloader_inference()] high_res shape: {hi_res_truth.shape}")
        # print(f"INFO [model_dataloader_inference()] low_res  shape: {low_res.shape}")
        
        low_res = low_res.to(device)
        hi_res_truth = hi_res_truth.to(device)
        
        optimizer.zero_grad()
        
        inference = model(low_res)
        
        # print(f"INFO [model_dataloader_inference()] inference shape: {inference.shape}")
        
        loss = criterion(inference, hi_res_truth)
        
        if(optimizer is not None):
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        # print(f"INFO LOSS ITEM: {loss.item() / len(batch)}")
    loss = running_loss / float(len(dataloader.dataset))
    return loss

def sec_to_human(seconds):
    """Return a number of seconds to hours, minutes, and seconds"""
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hours, minutes, seconds)

def normalize_tensor_image(rgb_tensor):
    """
    Expects rgb_tensor to be of shape (C, H, W)
    """
    
    # rgb_tensor = rgb_tensor.permute(2, 1, 0) # From (C,H,W) to (H,W,C)
    rgb_min = rgb_tensor.min()
    rgb_max = rgb_tensor.max()

    # For every pixel
    # for i in range(len(rgb_tensor)):
    #     for j in range(len(rgb_tensor[0])):
    #         largest = torch.max(rgb_tensor[i,j])
    #         smallest = torch.min(rgb_tensor[i,j])
    #         if(largest > 1.0 or smallest < 0):
                # rgb_tensor[i,j] -= smallest
                # rgb_tensor[i,j] /= (largest - smallest)
                # print(f"HAHAH I FOUND ONE:", rgb_tensor[i,j])
                # rgb_tensor[i,j] /= largest
                # rgb_tensor[i,j] = (rgb_tensor[i,j] - rgb_min) / (rgb_max - rgb_min)
    # rgb_tensor = rgb_tensor.permute(2, 1, 0) # From (H,W,C) to (C,H,W)
    
    rgb_tensor = (rgb_tensor - rgb_min) / (rgb_max - rgb_min)
                
    return rgb_tensor


def plot_inference_times():
    inference = None
    truths = None
    
    batch_sizes = np.arange(2000) + 2
    inference_times = np.zeros(2000)
    
    low_res_tensors = torch.load('../data/data/low_res_tensors.pt', weights_only=True)
    
    n_avg = 100
    
    model.eval()
    for i in range(len(batch_sizes)):
        batch = low_res_tensors[0:batch_sizes[i]]
        
        batch = batch.to(device)
        
        for n in range(n_avg):
            t_prior = time.time()
            inference = model(batch)
            t_inference = time.time() - t_prior
            
            inference_times[i] += t_inference
        inference_times[i] /= n_avg
        
        print(f"INFO [inference.py] Batch size: {batch_sizes[i]:>5} Inference Time: {inference_times[i]:.6f}", flush=True)
        
        # loss = criterion(inference, hi_res_truth)

    plt.plot(batch_sizes, inference_times)
    plt.savefig('./inference_times.png')

if __name__ == '__main__':
    tstart = time.time()
    print(f"INFO [inference.py] Starting script at {tstart}")
    
    # Set up device, model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO [inference.py] Using device: {device} [torch version: {torch.__version__}]')
    print(f'INFO [inference.py] Python version: {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')
    model = FSRCNN(upscale_factor=2, color_space=COLOR_SPACE).to(device)
    model.load_state_dict(torch.load('./saved_weights/good_emulated_56.pth', weights_only=True))
    
    # print_model_summary(model, 1, 3, 32, 32)
    # exit()
    
    criterion = nn.MSELoss()
    
    # Get dataset
    seed = 50  # Set the seed for reproducibility
    # torch.manual_seed(seed)
    print("INFO [inference.py] Loading Tensor pair dataset")
    full_dataset = SR_tensor_dataset(high_res_tensors_path='../data/data/high_res_tensors_10k.pt', low_res_tensors_path='../data/data/low_res_tensors_10k.pt')
    
    # Create train and test datasets. Set small train set for faster training

    train_dataset, valid_dataset, test_dataset = \
            torch.utils.data.random_split(full_dataset, [0.85, 0.10, 0.05], generator=torch.Generator())
    num_train_samples = len(train_dataset)
    print(f'INFO [inference.py] Total num data samples:    {len(full_dataset)}')
    print(f'INFO [inference.py] Num of training samples:   {num_train_samples}')
    print(f'INFO [inference.py] Num of validation samples: {len(valid_dataset)}')
    print(f'INFO [inference.py] Num of test samples:       {len(test_dataset)}')
    
    # Get Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True)
    print(f'INFO [inference.py] Num training batches: {len(train_dataloader)}')
    #scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.5)

    low_res, hi_res_truth = next(iter(test_dataloader))
    
    inference = model(low_res.to(device))
    plot_images(low_res, normalize_tensor_image(inference), hi_res_truth)
    
    # loss = criterion(inference, hi_res_truth)

    tEnd = time.time()
    print(f"INFO [inference.py] Ending script. Took {tEnd-tstart:.2f} seconds.")
    print(f"INFO [inference.py] HH:MM:SS --> {sec_to_human(tEnd-tstart)}")
    
    