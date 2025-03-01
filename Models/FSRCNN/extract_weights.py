from FSRCNN import *
import sys
import time
import torch
import numpy as np

COLOR_SPACE = 'rgb'

def sec_to_human(seconds):
    """Return a number of seconds to hours, minutes, and seconds"""
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hours, minutes, seconds)


def print_c_arr(weight_mat:torch.tensor):
    for channel in range(len(weight_mat)):
        print('{')
        for row in range(len(weight_mat[0])):
            print('{', end='')
            for col in range(len(weight_mat[0][0])):
                print(f'{weight_mat[channel,row,col].numpy():>11.8f}, ', end='')
            print('},')
        print('},')
    
if __name__ == '__main__':
    tstart = time.time()
    
    # Set up device, model
    device = torch.device('cpu')
    print(f'INFO [extract_weights.py] Using device: {device} [torch version: {torch.__version__}]')
    print(f'INFO [extract_weights.py] Python version: {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')
    model = FSRCNN(upscale_factor=2, color_space=COLOR_SPACE).to(device)
    model.load_state_dict(torch.load('./saved_weights/example_vitis_hls_weights_44.pth', weights_only=True))
    
    state_dict = model.state_dict()
    
    # for param_tensor in state_dict:
    #     print(f"Parameter: {param_tensor}, Shape: {state_dict[param_tensor].size()}")
    
    # print(state_dict['feature_extraction.0.weight'].shape) # Conv2D weights
    # print(state_dict['feature_extraction.0.bias'].shape)   # Conv2D bias
    print(state_dict['feature_extraction.1.weight'])       # PReLU
    
    # weights_0 = state_dict['feature_extraction.0.weight'][0]
    # print(weights_0)
    # print_c_arr(weights_0)
    # exit()
    
    # print("Bias from feature extraction filter0: ")
    # print(state_dict['feature_extraction.0.bias'])
    # exit()
    # print(state_dict['feature_extraction.0.weight'][0]) # Conv2D weights

    global_min = float('inf')
    global_max = float('-inf')

    # Iterate over all parameters in the state_dict
    for param_name, param_tensor in state_dict.items():
        if 'weight' in param_name:  # Ensure we're only looking at weight tensors
            param_min = param_tensor.min().item()
            param_max = param_tensor.max().item()
            
            if param_min < global_min:
                global_min = param_min
                min_param_name = param_name
            
            if param_max > global_max:
                global_max = param_max
                max_param_name = param_name

    print(f"The smallest weight is {global_min:.6f} found in {min_param_name}")
    print(f"The largest weight is  {global_max:.6f} found in {max_param_name}")
    