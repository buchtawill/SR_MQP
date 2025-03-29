from FSRCNN import *
import sys
import time
import torch
import torchinfo
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

def print_weights_as_c_array(state_dict, tensor_name="feature_extraction.0.weight"):
    if tensor_name not in state_dict:
        print(f"Tensor {tensor_name} not found in state_dict.")
        return
    
    weights = state_dict[tensor_name]  # Shape: [44, 3, 5, 5]
    weights = weights.numpy()  # Convert to NumPy array
    shape = weights.shape
    
    print(f"const fixed_4_8_t weights[{shape[0]}][{shape[1]}][{shape[2]}][{shape[3]}] = {{")
    for i in range(shape[0]):
    # for i in range(1):
        print("    // Filter", i)
        print("    {")
        for j in range(shape[1]):
            print("        {   // Channel", j)
            for k in range(shape[2]):
                row_values = ", ".join(f"{weights[i, j, k, l]:11.8f}" for l in range(shape[3]))
                print(f"            {{ {row_values} }},")
            print("        },")
        print("    },")
    print("};")
    
def print_model_summary(model, batch_size, in_channels, height, width):
    torchinfo.summary(model, input_size=(batch_size, in_channels, height, width))
    
if __name__ == '__main__':
    tstart = time.time()
    
    # Set up device, model
    device = torch.device('cpu')
    model = FSRCNN(upscale_factor=2, color_space=COLOR_SPACE).to(device)
    model.load_state_dict(torch.load('./saved_weights/example_vitis_hls_weights_44.pth', weights_only=True))
    
    state_dict = model.state_dict()
    
    # print(state_dict['feature_extraction.0.weight'].shape) # Conv2D weights
    # print(state_dict['feature_extraction.0.bias'].shape)   # Conv2D bias
    # print(state_dict['feature_extraction.1.weight'])       # PReLU
    
    # weights_0 = state_dict['feature_extraction.0.weight'][0]
    # print(weights_0)
    # print_c_arr(weights_0)
    # exit()
    
    # print("Bias from feature extraction filter0: ")
    # print(state_dict['feature_extraction.0.bias'])
    # print(state_dict['feature_extraction.0.weight'][0]) # Conv2D weights

    global_min = float('inf')
    global_max = float('-inf')
    
    # print_weights_as_c_array(state_dict)
    # exit()

    # Iterate over all parameters in the state_dict
    for param_name, param_tensor in state_dict.items():
        
        print(param_name)
        # if 'weight' in param_name:  # Ensure we're only looking at weight tensors
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
    