import torch
import numpy as np
import os


def quantize_and_save_weights(weight_file_path, output_path):
    """
    Load weights and save only the quantized int8 values and scales.
    """
    # Load the TorchScript model
    model = torch.jit.load(weight_file_path)

    # Extract parameters
    quantized_data = {
        'weights': {},
        'scales': {}
    }

    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            # Convert to int8
            tensor_max = torch.max(torch.abs(param.data))
            scale = 127.0 / tensor_max
            quantized = torch.round(param.data * scale).to(torch.int8)

            # Store only the int8 values and scale
            quantized_data['weights'][name] = quantized
            quantized_data['scales'][name] = scale

    # Save only the essential data
    torch.save(quantized_data, output_path)

    # Print size comparison
    orig_size = os.path.getsize(weight_file_path) / (1024 * 1024)  # MB
    new_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"Original size: {orig_size:.2f} MB")
    print(f"Quantized size: {new_size:.2f} MB")

    return quantized_data


if __name__ == "__main__":
    weight_file = "fsrcnn_model.pt"
    output_file = "fsrcnn_int8.pth"  # Changed extension since it's not TorchScript anymore

    try:
        quantized_data = quantize_and_save_weights(weight_file, output_file)

        # Print some statistics
        for name, weights in quantized_data['weights'].items():
            print(f"\nLayer: {name}")
            print(f"Shape: {weights.shape}")
            print(f"Memory usage: {weights.numel() * weights.element_size()} bytes")

    except Exception as e:
        print(f"Error during quantization: {str(e)}")