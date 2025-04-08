#!/usr/bin/env python
import argparse
import numpy as np
import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description="Convolution Reference using PyTorch with saved weights")
    parser.add_argument('--mode', type=str, default="axi",
                        help='Data mode: "axi" for uint8 input, "fifo" for float32 input')
    parser.add_argument('--input', type=str, required=True,
                        help='Input binary file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output binary file (raw float32)')
    parser.add_argument('--in_channels', type=int, required=True,
                        help='Number of input channels')
    parser.add_argument('--out_channels', type=int, required=True,
                        help='Number of output channels')
    parser.add_argument('--kernel_size', type=int, required=True,
                        help='Kernel size (assumed square)')
    parser.add_argument('--width', type=int, required=True,
                        help='Input image width')
    parser.add_argument('--height', type=int, required=True,
                        help='Input image height')
    parser.add_argument('--conv_name', type=str, required=True,
                        help='Convolution layer name (e.g., "feature_extraction.0")')
    parser.add_argument('--stride', type=int, required=True,
                        help='Stride for convolution')
    parser.add_argument('--padding', type=int, required=True,
                        help='Padding for convolution')
    args = parser.parse_args()

    # Read input file based on mode.
    if args.mode == "fifo":
        try:
            data = np.fromfile(args.input, dtype=np.float32)
        except Exception as e:
            print("Error reading input file:", e)
            exit(1)
    else:  # axi mode: uint8 input
        try:
            data = np.fromfile(args.input, dtype=np.uint8)
        except Exception as e:
            print("Error reading input file:", e)
            exit(1)
    expected_size = args.height * args.width * args.in_channels
    if data.size != expected_size:
        print("Input data size does not match expected dimensions. Got", data.size, "expected", expected_size)
        exit(1)
    if args.mode == "fifo":
        input_data = data.reshape((args.height, args.width, args.in_channels)).astype(np.float32)
    else:
        input_data = data.reshape((args.height, args.width, args.in_channels)).astype(np.float32)
        if args.conv_name.startswith("feature_extraction"):
            input_data = input_data / 255.0
        else:
            input_data = (2.0 * input_data / 255.0) - 1.0

    # Debug: print last 50 elements of input_data (flattened)
    flat_input = input_data.flatten()
    print("Python - First 50 elements of input_data:")
    print(flat_input[:50])

    # Rearrange to (N, C, H, W) for pyTorch std
    input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).unsqueeze(0)

    # Load saved weights
    weights_file = '/home/dmaljabbari/Desktop/SR_MQP/Models/FSRCNN/saved_weights/example_vitis_hls_weights_44.pth'
    try:
        state_dict = torch.load(weights_file, map_location=torch.device('cpu'))
    except Exception as e:
        print("Error loading weights file:", e)
        exit(1)

    # Extract convolution weights and bias
    weight_key = args.conv_name + ".weight"
    bias_key = args.conv_name + ".bias"
    if weight_key not in state_dict:
        print("Error: Weight for", args.conv_name, "not found in the weights file.")
        exit(1)
    conv_weight = state_dict[weight_key]
    conv_bias = state_dict[bias_key] if bias_key in state_dict else None

    print("Convolution weight shape:", conv_weight.shape)
    if conv_bias is not None:
        print("Convolution bias shape:", conv_bias.shape)

    # Create convolution layer
    if args.conv_name.startswith("deconv"):
        in_ch = conv_weight.size(0)
        out_ch = conv_weight.size(1)
        k_size = conv_weight.size(2)  # square kernel
        conv_layer = nn.ConvTranspose2d(in_ch, out_ch, k_size, stride=args.stride, 
                                        padding=args.padding, output_padding=1, bias=(conv_bias is not None))
    else:
        in_ch = conv_weight.size(1)
        out_ch = conv_weight.size(0)
        k_size = conv_weight.size(2)
        conv_layer = nn.Conv2d(in_ch, out_ch, k_size, stride=args.stride, padding=args.padding, bias=(conv_bias is not None))
    
    with torch.no_grad():
        conv_layer.weight.copy_(conv_weight)
        if conv_bias is not None:
            conv_layer.bias.copy_(conv_bias)

    # Manually apply PReLU activation for all layers but deconv
    prelu_param = None
    if args.conv_name.startswith("expand"): # Change layer name here 
        prelu_key = "expand.1.weight" # Change the layer's preLu key here
        if prelu_key in state_dict:
            prelu_param = state_dict[prelu_key]
            print("Loaded PReLU parameter from", prelu_key, "with shape", prelu_param.shape)
        else:
            print("Warning: PReLU parameter", prelu_key, "not found. No PReLU activation will be applied.")

    # Perform convolution.
    with torch.no_grad():
        output = conv_layer(input_tensor)
    # At this point, bias has been added (if any).

    # If applicable, apply the PReLU activation manually.
    if prelu_param is not None:
        # If prelu_param is a single scalar, or a vector of per-channel parameters.
        if prelu_param.numel() == 1:
            alpha = prelu_param.item()
            output = torch.where(output >= 0, output, output * alpha)
            print("single scalar or vector per channel")
        else:
            # Reshape alpha to (1, C, 1, 1) and broadcast.
            alpha = prelu_param.view(1, -1, 1, 1)
            output = torch.where(output >= 0, output, output * alpha)
    
    # Write raw float32 output (without clamping or rounding)
    output_np = output.cpu().numpy().astype(np.float32)
    flat_output = output.flatten()
    # print("Python - First 50 elements of output:")
    # print(flat_output[:50])

    try:
        output_np.tofile(args.output)
    except Exception as e:
        print("Error writing output file:", e)
        exit(1)

if __name__ == '__main__':
    main()
