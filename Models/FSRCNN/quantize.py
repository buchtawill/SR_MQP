from FSRCNN import *
import sys
import time
import torch
# import numpy as np
# from tqdm import tqdm
# from PIL import Image
import torch.nn as nn
import torch.quantization
# import torch.optim as optim
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from low_hi_res_dataset import SR_tensor_dataset

BATCH_SIZE = 128

if __name__ == '__main__':
    tstart = time.time()
    print(f"INFO [quantize.py] Starting script at {tstart}")
    
    #Set up device, model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO [quantize.py] Using device: {device} [torch version: {torch.__version__}]')
    print(f'INFO [quantize.py] Python version: {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')
    model = FSRCNN(upscale_factor=2, color_space='rgb').to(device)
    model.load_state_dict(torch.load('./saved_weights/100E_5em4_b64.pth', weights_only=True))
    
    seed = 50  # Set the seed for reproducibility
    torch.manual_seed(seed)
    print("INFO [quantize.py] Loading Tensor pair dataset")
    t_before = time.time()
    full_dataset = SR_tensor_dataset(high_res_tensors_path='../data/data/high_res_tensors.pt', low_res_tensors_path='../data/data/low_res_tensors.pt')
    print(f'INFO [quantize.py] It took {time.time() - t_before : .2f} seconds to load the data')
    
    train_dataset, valid_dataset, test_dataset = \
            torch.utils.data.random_split(full_dataset, [0.85, 0.10, 0.05], generator=torch.Generator())
    num_train_samples = len(train_dataset)
    print(f'INFO [quantize.py] Total num data samples:    {len(full_dataset)}')
    print(f'INFO [quantize.py] Num of training samples:   {num_train_samples}')
    print(f'INFO [quantize.py] Num of validation samples: {len(valid_dataset)}')
    print(f'INFO [quantize.py] Num of test samples:       {len(test_dataset)}')
    
    # Get Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True)
    print(f'INFO [quantize.py] Num training batches: {len(train_dataloader)}', flush = True)
    
    model.eval()
    
    # model_fused = torch.quantization.fuse_modules(model, [['conv2', 'prelu']])
    
    model_prepared = torch.quantization.prepare(model)
    
    for low_res, high_res in train_dataloader:
        low_res *= 255
        print(low_res[0])
        exit()
        model_prepared(low_res)
        
    model_quantized = torch.quantization.convert(model_prepared)
    
    # Now compare quantized model to normal model