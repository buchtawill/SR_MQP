#!/usr/bin/env python
import argparse
import numpy as np
import torch
import torch.nn as nn

class FullCNN(nn.Module):
    def __init__(self, upscale_factor):
        super(FullCNN, self).__init__()
        # Feature extraction layer: Conv2d(3,44,5x5, stride=1, padding=2) + PReLU(44)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.PReLU(16)
        )
        # Shrinking layer: Conv2d(44,12,1x1) + PReLU(12)
        self.shrink = nn.Sequential(
            nn.Conv2d(16, 12, kernel_size=1, stride=1, padding=0),
            nn.PReLU(12)
        )
        # Mapping layer: 3 layers of Conv2d(12,12,3x3, stride=1, padding=1) + PReLU(12)
        self.map = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(12),
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(12),
            nn.Conv2d(12, 8, kernel_size=3, stride=1, padding=1),
            nn.PReLU(8)
        )
        # Expanding layer: Conv2d(12,44,1x1) + PReLU(44)
        self.expand = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),
            nn.PReLU(8)
        )
        # Deconvolution layer: ConvTranspose2d(44,3,9x9, stride=upscale_factor, padding=4, output_padding=upscale_factor-1)
        self.deconv = nn.ConvTranspose2d(8, 3, kernel_size=7, stride=upscale_factor, padding=3, output_padding=upscale_factor-1)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x

def main():
    parser = argparse.ArgumentParser(description="Full CNN Predictor using PyTorch with saved weights")
    parser.add_argument('--mode', type=str, default="axi",
                        help='Data mode: "axi" expects uint8 input')
    parser.add_argument('--input', type=str, required=True,
                        help='Input binary file (raw uint8 image)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output binary file (raw float32)')
    parser.add_argument('--width', type=int, required=True,
                        help='Input image width')
    parser.add_argument('--height', type=int, required=True,
                        help='Input image height')
    parser.add_argument('--channels', type=int, required=True,
                        help='Number of input channels (should be 3 for RGB)')
    parser.add_argument('--upscale_factor', type=int, required=True,
                        help='Upscale factor for deconvolution')
    args = parser.parse_args()

    # Read input file
    try:
        data = np.fromfile(args.input, dtype=np.uint8)
    except Exception as e:
        print("Error reading input file:", e)
        exit(1)
    expected_size = args.width * args.height * args.channels
    if data.size != expected_size:
        print("Input data size mismatch. Got", data.size, "expected", expected_size)
        exit(1)
    # Reshape to (H, W, C)
    data = data.reshape((args.height, args.width, args.channels)).astype(np.float32)
    # Scale input: divide by 256 [0, ~1]
    input_scaled = data / 256.0

    # Debug print: print first 50 elements of flattened input.
    flat_input = input_scaled.flatten()
    print("Python - First 50 elements of scaled input:")
    print(flat_input[:50])

    # Rearrange to (N, C, H, W)
    input_tensor = torch.from_numpy(input_scaled).permute(2, 0, 1).unsqueeze(0)

    # Instantiate full CNN and load weights.
    model = FullCNN(upscale_factor=args.upscale_factor)
    weights_file = "/home/dmaljabbari/Desktop/SR_MQP/Models/FSRCNN/saved_weights/weights_nerfed.pth"
    try:
        state_dict = torch.load(weights_file, map_location=torch.device('cpu'))
    except Exception as e:
        print("Error loading weights file:", e)
        exit(1)
    model.load_state_dict(state_dict)
    model.eval()

    # Perform forward pass.
    with torch.no_grad():
        output = model(input_tensor)
    # Multiply output by 256 to RGB scale.
    output = output * 256.0
    # Clip the output: any value below 0 becomes 0; above 255 becomes 255.
    output = torch.clamp(output, 0.0, 255.0)
    # Rearrange to (H, W, C)
    output = output.squeeze(0).permute(1, 2, 0)
    output_np = output.cpu().numpy().astype(np.float32)

    # (Optional) print first 10 output values for debugging.
    flat_output = output_np.flatten()
    print("Python - First 10 elements of output:")
    print(flat_output[:10])

    # Write output as raw float32.
    try:
        output_np.tofile(args.output)
    except Exception as e:
        print("Error writing output file:", e)
        exit(1)

if __name__ == '__main__': 
    main()
