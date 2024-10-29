# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Realize the model definition function."""
import os
from math import sqrt

import torch
from torch import nn
import sys
from PIL import Image
import torchvision.transforms as transforms


class FSRCNN(nn.Module):
    """

    Args:
        upscale_factor (int): Image magnification factor.
        color_space (str): 'yuv' or 'rgb'. Expects YUV tensor to have luminance as first channel
    """

    def __init__(self, upscale_factor: int, color_space = 'rgb') -> None:
        
        mid_f_maps = 56
        
        super(FSRCNN, self).__init__()
        
        self.color_space = color_space
        self.input_channels = 3 if (color_space == 'rgb') else 1
        
        print("INFO [FSRCNN.py::__init__()] Num input channels:", self.input_channels)
        
        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(self.input_channels, mid_f_maps, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(mid_f_maps)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(mid_f_maps, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )

        # Mapping layer.
        # 'm' mapping layers - paper uses 4 for best results
        self.map = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(12, mid_f_maps, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(mid_f_maps)
        )

        # Deconvolution layer.
        self.deconv = nn.ConvTranspose2d(mid_f_maps, self.input_channels, (9, 9), (upscale_factor, upscale_factor), (4, 4), (upscale_factor - 1, upscale_factor - 1))

        #nearest or bilinear
        self.simple_upscale = nn.Upsample(scale_factor=upscale_factor, mode='nearest')

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if(self.color_space == 'rgb'):
            return self._forward_impl_rgb(x)

        elif(self.color_space == 'yuv'):
            return self._forward_impl_yuv(x)
        
        else:
            raise Exception("ERROR [FSRCNN::forward()] Unkown color space:", self.color_space)
        
    # Support torch.script function.
    def _forward_impl_yuv(self, x: torch.Tensor) -> torch.Tensor:
        
        # x is of shape [batch size, channels, H, W]
        # channels[0] is Y, which is what we want to upscale
        y_channel_only = x[:, 0, :, :]                 # shape = [batch size, 1, H, W]
        y_channel_only = y_channel_only.unsqueeze(1)   # shape = [batch size, 1, H, W]
        uv_channels    = x[:, 1:, :, :]
        # u_channel_only = x[:, 1, :, :]
        # v_channel_only = x[:, 2, :, :]
        
        out_y = self.feature_extraction(y_channel_only)
        out_y = self.shrink(out_y)
        out_y = self.map(out_y)
        out_y = self.expand(out_y)
        out_y = self.deconv(out_y)
        
        upscaled_uv = self.simple_upscale(uv_channels)
        
        merged_yuv = torch.cat((out_y, upscaled_uv), dim=1)

        return merged_yuv
    
    # Support torch.script function.
    def _forward_impl_rgb(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and standard deviation initialized by random extraction 0.001 (deviation is 0).
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)


def process_image(input_path):
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FSRCNN(upscale_factor=2, color_space='rgb').to(device)

    # Load pretrained weights
    weights_path = '100E_5em4_b64_CPU.pth'  # Update this path to your weights file
    if not os.path.exists(weights_path):
        raise Exception(f"Model weights not found at {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print(f"Loaded model weights from {weights_path}")

    # Load and process image
    img = Image.open(input_path).convert('RGB')
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)

    print(f"Processing image of size: {img.size}")
    print(f"Input tensor shape: {input_tensor.shape}")

    # Process through model
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Output tensor shape: {output.shape}")
    print(f"Output tensor range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Save output
    output = output.cpu().squeeze(0).clamp(0, 1)
    output_img = transforms.ToPILImage()(output)
    output_img.save('upscaled_image.png', 'PNG')
    print(f"Saved output image of size: {output_img.size}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python FSRCNN.py <input_image>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    process_image(input_path)