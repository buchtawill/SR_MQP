import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from Models.FSRCNN.FSRCNN import FSRCNN
import os
import matplotlib.pyplot as plt


def get_model_size_info(model):
    """Get size information for model parameters"""
    size_info = {}
    total_size = 0
    for name, param in model.named_parameters():
        param_size = param.nelement() * param.element_size()
        size_info[name] = {
            'shape': param.shape,
            'dtype': param.dtype,
            'size_bytes': param_size,
            'size_kb': param_size / 1024
        }
        total_size += param_size
    size_info['total_kb'] = total_size / 1024
    return size_info


def load_model(model_path: str):
    """Load model from either .pt or .pth file"""
    print(f"\nAnalyzing model: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / 1024:.2f} KB")

    if model_path.endswith('.pt'):
        model = torch.jit.load(model_path)
        test_input = torch.randn(1, 3, 100, 100)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"Model scaling test - Input: {test_input.shape}, Output: {test_output.shape}")
        size_info = get_model_size_info(model)
    elif model_path.endswith('.pth'):
        quantized_data = torch.load(model_path)
        model = FSRCNN(upscale_factor=2)

        test_input = torch.randn(1, 3, 100, 100)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"Model scaling test - Input: {test_input.shape}, Output: {test_output.shape}")

        print("\nQuantized weights information:")
        total_quantized_size = 0
        for name in quantized_data['weights']:
            weight = quantized_data['weights'][name]
            size_bytes = weight.nelement() * weight.element_size()
            print(f"\n{name}:")
            print(f"  Shape: {weight.shape}")
            print(f"  Dtype: {weight.dtype}")
            print(f"  Size: {size_bytes / 1024:.2f} KB")
            total_quantized_size += size_bytes
        print(f"\nTotal quantized weights size: {total_quantized_size / 1024:.2f} KB")

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in quantized_data['weights']:
                    quantized = quantized_data['weights'][name]
                    scale = quantized_data['scales'][name]
                    param.data = quantized.float() / scale

    else:
        raise ValueError(f"Unsupported file extension for {model_path}")

    size_info = get_model_size_info(model)
    print("\nModel parameters information:")
    for name, info in size_info.items():
        if name != 'total_kb':
            print(f"\n{name}:")
            print(f"  Shape: {info['shape']}")
            print(f"  Dtype: {info['dtype']}")
            print(f"  Size: {info['size_kb']:.2f} KB")
    print(f"\nTotal parameters size: {size_info['total_kb']:.2f} KB")

    return model


def create_test_output(model, input_image_path, output_name):
    """Process image through model and save result"""
    img = Image.open(input_image_path)
    print(f"Input image size: {img.size}")

    if img.mode != 'RGB':
        img = img.convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0)
    print(f"Input tensor size: {img_tensor.shape}")

    with torch.no_grad():
        output = model(img_tensor)
        print(f"Output tensor size: {output.shape}")

    output_img = transforms.ToPILImage()(output.squeeze(0).clamp(0, 1))
    print(f"Output image size: {output_img.size}")

    output_img.save(output_name)
    print(f"Saved output image as: {output_name}")
    return output_img


def create_detailed_comparison(input_img_path, output_img_path):
    """Create a detailed comparison between input and upscaled output images"""
    input_img = Image.open(input_img_path)
    output_img = Image.open(output_img_path)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])

    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[1, 3])

    ax1.imshow(input_img)
    ax2.imshow(output_img)

    ax1.set_title(f'Input Image\nSize: {input_img.size[0]}x{input_img.size[1]} px', pad=20)
    ax2.set_title(f'Upscaled Output\nSize: {output_img.size[0]}x{output_img.size[1]} px', pad=20)

    x_center = input_img.size[0] // 2
    y_center = input_img.size[1] // 2
    zoom_size = min(100, min(input_img.size) // 4)

    x1, y1 = x_center - zoom_size, y_center - zoom_size
    x2, y2 = x_center + zoom_size, y_center + zoom_size

    ax3.imshow(input_img)
    ax3.set_xlim(x1, x2)
    ax3.set_ylim(y2, y1)
    ax4.imshow(input_img)
    ax4.set_xlim(x1 - zoom_size // 2, x1 + zoom_size // 2)
    ax4.set_ylim(y1 + zoom_size // 2, y1 - zoom_size // 2)

    scale_factor = output_img.size[0] / input_img.size[0]
    ax5.imshow(output_img)
    ax5.set_xlim(x1 * scale_factor, x2 * scale_factor)
    ax5.set_ylim(y2 * scale_factor, y1 * scale_factor)
    ax6.imshow(output_img)
    ax6.set_xlim((x1 - zoom_size // 2) * scale_factor, (x1 + zoom_size // 2) * scale_factor)
    ax6.set_ylim((y1 + zoom_size // 2) * scale_factor, (y1 - zoom_size // 2) * scale_factor)

    ax3.set_title('Input Zoom Region 1')
    ax4.set_title('Input Zoom Region 2')
    ax5.set_title('Output Zoom Region 1')
    ax6.set_title('Output Zoom Region 2')

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axis('off')

    resolution_increase = (output_img.size[0] * output_img.size[1]) / (input_img.size[0] * input_img.size[1])
    fig.suptitle(f'Resolution Comparison\nTotal pixel increase: {resolution_increase:.2f}x',
                 fontsize=16, y=0.98)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('detailed_comparison.png', bbox_inches='tight', dpi=300)
    print("Saved detailed comparison as 'detailed_comparison.png'")

    print("\nImage Statistics:")
    print(f"Input resolution: {input_img.size[0]}x{input_img.size[1]} ({input_img.size[0] * input_img.size[1]} pixels)")
    print(
        f"Output resolution: {output_img.size[0]}x{output_img.size[1]} ({output_img.size[0] * output_img.size[1]} pixels)")
    print(f"Resolution increase: {resolution_increase:.2f}x")
    print(f"Scale factor: {scale_factor:.2f}x per dimension")


def compare_outputs(img_path):
    """Create and save comparison of original and quantized model outputs"""
    print("\nLoading models for comparison...")


    model = load_model("fsrcnn_model.pt")
    create_test_output(model, "raw_wiiframe.png", "upscaled_output.png")
    create_detailed_comparison("raw_wiiframe.png", "upscaled_output.png")
    quantized_model = load_model("fsrcnn_int8.pth")

    print("\nProcessing with original model...")
    original_output = create_test_output(model, img_path, "original_output.png")

    print("\nProcessing with quantized model...")
    quantized_output = create_test_output(quantized_model, img_path, "quantized_output.png")

    create_detailed_comparison(img_path, "original_output.png")

    mse = np.mean((np.array(original_output) - np.array(quantized_output)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    print(f"\nPSNR between original and quantized outputs: {psnr:.2f} dB")

    mse = np.mean((np.array(original_output) - np.array(quantized_output)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    print(f"PSNR between original and quantized outputs: {psnr:.2f} dB")


def test_model(model_path: str, test_image_path: str = "raw_wiiframe.png"):
    """Test the model on an image"""
    print(f"\nTesting model: {model_path}")

    try:
        model = load_model(model_path)
        print("Model loaded successfully")

        test_input = torch.randn(1, 3, 28, 28)
        with torch.no_grad():
            test_output = model(test_input)
            print(f"Random input test successful.")
            print(f"Test input shape: {test_input.shape}")
            print(f"Test output shape: {test_output.shape}")
            print(f"Scaling factor: {test_output.shape[-1] / test_input.shape[-1]:.2f}x")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:

        output_img = create_test_output(model, test_image_path, "test_output.png")
        print(f"Final output image size: {output_img.size}")

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    print("\nGenerating comparison of original and quantized models...")
    compare_outputs("raw_wiiframe.png")

    print("\nTesting original model...")
    test_model("fsrcnn_model.pt")

    print("\nTesting quantized model...")
    test_model("fsrcnn_int8.pth")