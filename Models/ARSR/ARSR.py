import torch
import torch.nn as nn


class OverparameterizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, expanded_channels=None):
        super().__init__()
        self.training_mode = True
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Typically expanded_channels is larger than both in_channels and out_channels
        if expanded_channels is None:
            expanded_channels = max(in_channels, out_channels) * 2

        # Training convolutions (expanded)
        self.conv1 = nn.Conv2d(in_channels, expanded_channels, kernel_size,
                               stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(expanded_channels, out_channels, 1)

        # Inference convolution (collapsed)
        self.collapsed_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding)

    def forward(self, x):
        if self.training_mode:
            # Training path
            out = self.conv1(x)
            out = self.conv2(out)
        else:
            # Inference path
            out = self.collapsed_conv(x)
        return out

    def collapse_weights(self):
        """Collapse the two-layer convolution into a single layer for inference"""
        with torch.no_grad():
            # Get the shapes of the convolution kernels
            out_channels, in_channels, kh, kw = self.conv1.weight.shape

            # Reshape conv1 weights to prepare for collapse
            conv1_weights = self.conv1.weight.view(out_channels, in_channels * kh * kw)

            # Reshape conv2 weights
            conv2_weights = self.conv2.weight.squeeze(-1).squeeze(-1)

            # Compute collapsed weights
            collapsed = torch.mm(conv2_weights, conv1_weights)
            collapsed = collapsed.view(self.out_channels, self.in_channels, kh, kw)

            # Update collapsed convolution weights
            self.collapsed_conv.weight.data.copy_(collapsed)

            # Handle bias terms
            if self.conv1.bias is not None and self.conv2.bias is not None:
                # Compute combined bias
                bias = torch.mm(conv2_weights, self.conv1.bias.unsqueeze(1)).squeeze() + self.conv2.bias
                self.collapsed_conv.bias.data.copy_(bias)

    def train(self, mode=True):
        """Switch between training and inference modes"""
        super().train(mode)
        self.training_mode = mode
        if not mode:
            self.collapse_weights()

class ARSRNetwork(nn.Module):
    def __init__(self, scale_factor=2, n_feature_layers=6, n_mapping_layers=6, in_channels=3):
        super().__init__()
        self.scale_factor = scale_factor

        # Initial feature extraction
        self.first_conv = OverparameterizedConv2d(in_channels, 64, kernel_size=3, padding=1)

        # Feature extraction layers
        self.feature_layers = nn.ModuleList([
            OverparameterizedConv2d(64, 64, kernel_size=3, padding=1)
            for _ in range(n_feature_layers)
        ])

        # Non-linear mapping layers
        self.mapping_layers = nn.ModuleList([
            OverparameterizedConv2d(64, 64, kernel_size=3, padding=1)
            for _ in range(n_mapping_layers)
        ])

        # Final convolution before pixel shuffle
        self.final_conv = OverparameterizedConv2d(64, self.scale_factor ** 2 * in_channels, kernel_size=3, padding=1)

        # Pixel shuffle layer for upscaling
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Store input for global residual
        input_img = x

        # Initial convolution
        x = self.relu(self.first_conv(x))

        # Store for feature mapping residual
        feature_residual = x

        # Feature extraction
        for layer in self.feature_layers:
            x = self.relu(layer(x))

        # Non-linear mapping with residual
        mapping_input = x
        for layer in self.mapping_layers:
            x = self.relu(layer(x))
        x = x + mapping_input  # Residual connection

        # Final convolution and upscaling
        x = self.final_conv(x)
        x = self.pixel_shuffle(x)

        # Upsample the input for global residual connection
        upscaled_input = nn.functional.interpolate(
            input_img, scale_factor=self.scale_factor, mode='bicubic', align_corners=False
        )

        # Add global residual connection
        x = x + upscaled_input

        return x


# Example usage:
def create_model(scale_factor=2, n_feature_layers=6, n_mapping_layers=6):
    model = ARSRNetwork(scale_factor=scale_factor,
                        n_feature_layers=n_feature_layers,
                        n_mapping_layers=n_mapping_layers)
    return model


# Example for creating 2x and 4x models
model_2x = create_model(scale_factor=2)
model_4x = create_model(scale_factor=4)


# Example for processing a batch of images
def process_image(model, image):
    """
    Process a single image through the model
    image: tensor of shape (1, 1, H, W) - single channel Y component
    """
    model.eval()  # Set to evaluation mode (uses collapsed convolutions)
    with torch.no_grad():
        output = model(image)
    return output