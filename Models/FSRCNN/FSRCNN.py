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
from math import sqrt

import torch
from torch import nn


class FSRCNN(nn.Module):
    """
    Fast Super-Resolution CNN for RGB images.

    Args:
        upscale_factor (int): Image magnification factor.
    """

    def __init__(self, upscale_factor: int) -> None:
        super(FSRCNN, self).__init__()

        # RGB cuz fuck YUV
        self.input_channels = 3

        # Feature extraction layer parameters
        in_channels_feature = self.input_channels
        out_channels_feature = 16

        # Feature extraction layer
        self.feature_extraction = nn.Sequential(
            #                                    Kernel  stride  padding
            nn.Conv2d(in_channels_feature, out_channels_feature, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(out_channels_feature)
        )

        # Shrinking layer parameters
        in_channels_shrink = 16
        out_channels_shrink = 12

        # Shrinking layer
        self.shrink = nn.Sequential(
            nn.Conv2d(in_channels_shrink, out_channels_shrink, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(out_channels_shrink)
        )

        # Mapping layer parameters
        in_channels_map0 = 12
        out_channels_map0 = 12

        in_channels_map4 = 12
        out_channels_map4 = 12

        in_channels_map6 = 12
        out_channels_map6 = 8

        # Mapping - using map0, map4, and map6 from the image
        self.map = nn.Sequential(
            # Map0
            nn.Conv2d(in_channels_map0, out_channels_map0, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(out_channels_map0),

            # Map4
            nn.Conv2d(in_channels_map4, out_channels_map4, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(out_channels_map4),

            # Map6
            nn.Conv2d(in_channels_map6, out_channels_map6, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(out_channels_map6)
        )

        # Expanding layer parameters
        in_channels_expand = 8
        out_channels_expand = 8

        # Expanding layer
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels_expand, out_channels_expand, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(out_channels_expand)
        )

        # Deconvolution layer parameters
        in_channels_deconv = 8
        out_channels_deconv = self.input_channels

        # 7x7 Bernel (buchta kernel)
        self.deconv = nn.ConvTranspose2d(in_channels_deconv, out_channels_deconv, (7, 7),
                                         (upscale_factor, upscale_factor),
                                         (3, 3),
                                         (upscale_factor - 1, upscale_factor - 1))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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