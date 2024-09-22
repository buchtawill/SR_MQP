import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import math


def calculate_tiles(width: int, aspect_ratio: tuple, min_tile_size: int = 32) -> dict:
    height = int(width / (aspect_ratio[0] / aspect_ratio[1]))

    best_tile_size = 0
    best_tile_count = 0

    for tile_size in range(min_tile_size, min(width, height) + 1):
        if width % tile_size == 0 and height % tile_size == 0:
            horizontal_tiles = width // tile_size
            vertical_tiles = height // tile_size
            total_tiles = horizontal_tiles * vertical_tiles

            if total_tiles > best_tile_count:
                best_tile_size = tile_size
                best_tile_count = total_tiles

    if best_tile_size == 0:
        raise ValueError("No valid tile size found")

    return {
        "tile_size": best_tile_size,
        "tile_count": best_tile_count,
        "tiles_horizontal": width // best_tile_size,
        "tiles_vertical": height // best_tile_size
    }


class ImageDataset(Dataset):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png'))]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            *([transform] if transform else [])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.input_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]


def crop_tiles(input_dir: str, output_dir: str, tile_size: int = 64):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            image = Image.open(input_path)
            width, height = image.size

            # Calculate the number of tiles needed to cover the image
            num_tiles_h = (width + tile_size - 1) // tile_size
            num_tiles_v = (height + tile_size - 1) // tile_size

            for i in range(num_tiles_v):
                for j in range(num_tiles_h):
                    # Calculate tile coordinates with potential overlap
                    left = min(j * tile_size, width - tile_size)
                    top = min(i * tile_size, height - tile_size)
                    right = left + tile_size
                    bottom = top + tile_size

                    # Crop the tile
                    tile = image.crop((left, top, right, bottom))

                    tile_filename = f"{os.path.splitext(filename)[0]}_tile_{i}_{j}.png"
                    tile.save(os.path.join(output_dir, tile_filename))

            # Save tile info
            tile_info = {
                'original_size': (width, height),
                'tile_size': tile_size,
                'num_tiles': (num_tiles_h, num_tiles_v)
            }
            with open(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_info.txt"), 'w') as f:
                f.write(str(tile_info))

    print(f"Tiles saved in {output_dir}")


def downscale_cropped_tiles(input_dir: str, output_dir: str):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith('.png') and 'tile' in filename:
            input_path = os.path.join(input_dir, filename)

            # Create the new filename
            new_filename = f"{os.path.splitext(filename)[0]}_downscaled.png"
            output_path = os.path.join(output_dir, new_filename)

            try:
                with Image.open(input_path) as img:
                    # Calculate new dimensions (half of the original)
                    new_width = img.width // 2
                    new_height = img.height // 2

                    # Resize and save the image
                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                    resized_img.save(output_path)
                print(f"Saved downscaled image: {new_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Copy tile info files
    for filename in os.listdir(input_dir):
        if filename.endswith('_info.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
                f_out.write(f_in.read())


def repatch_downscaled_tiles(downscaled_dir, input_dir):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repatched_dir = os.path.join(current_dir, '1280_16x9_repatched')
    os.makedirs(repatched_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            original_name = os.path.splitext(filename)[0]

            # Load tile info
            info_file = os.path.join(downscaled_dir, f"{original_name}_info.txt")
            if not os.path.exists(info_file):
                print(f"Tile info not found for {original_name}")
                continue
            with open(info_file, 'r') as f:
                tile_info = eval(f.read())

            original_width, original_height = tile_info['original_size']
            tile_size = tile_info['tile_size'] // 2  # Downscaled tile size
            num_tiles_h, num_tiles_v = tile_info['num_tiles']

            # Create numpy arrays to hold the repatched image and weights
            repatched_array = np.zeros((original_height // 2, original_width // 2, 3), dtype=np.float32)
            weight_array = np.zeros((original_height // 2, original_width // 2), dtype=np.float32)

            for i in range(num_tiles_v):
                for j in range(num_tiles_h):
                    tile_filename = f"{original_name}_tile_{i}_{j}_downscaled.png"
                    tile_path = os.path.join(downscaled_dir, tile_filename)

                    if os.path.exists(tile_path):
                        tile = Image.open(tile_path)
                        tile_array = np.array(tile, dtype=np.float32) / 255.0

                        # Calculate position to paste the tile
                        left = min(j * tile_size, repatched_array.shape[1] - tile_size)
                        top = min(i * tile_size, repatched_array.shape[0] - tile_size)

                        # Create a weight mask for smooth blending
                        weight_mask = np.ones((tile_size, tile_size), dtype=np.float32)
                        weight_mask = np.minimum(weight_mask, np.arange(tile_size) + 1)
                        weight_mask = np.minimum(weight_mask, np.arange(tile_size)[:, np.newaxis] + 1)
                        weight_mask = np.minimum(weight_mask, np.arange(tile_size)[::-1] + 1)
                        weight_mask = np.minimum(weight_mask, np.arange(tile_size)[::-1][:, np.newaxis] + 1)

                        # Add the tile to the repatched array with blending
                        repatched_array[top:top + tile_size, left:left + tile_size] += tile_array * weight_mask[:, :,
                                                                                                    np.newaxis]
                        weight_array[top:top + tile_size, left:left + tile_size] += weight_mask

            # Normalize the repatched array
            weight_array = np.maximum(weight_array, 1e-6)[:, :, np.newaxis]
            repatched_array /= weight_array
            repatched_array = np.clip(repatched_array * 255, 0, 255).astype(np.uint8)

            # Convert the numpy array back to an image and save
            repatched_img = Image.fromarray(repatched_array)
            repatched_img.save(os.path.join(repatched_dir, f"{original_name}_repatched.png"))
            print(f"Saved repatched image: {original_name}_repatched.png")

def create_vertical_comparison(input_dir, repatched_dir, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png')):
            # Open original image
            original_path = os.path.join(input_dir, filename)
            original_img = Image.open(original_path)

            # Find corresponding repatched image
            repatched_filename = f"{os.path.splitext(filename)[0]}_repatched.png"
            repatched_path = os.path.join(repatched_dir, repatched_filename)

            if os.path.exists(repatched_path):
                repatched_img = Image.open(repatched_path)

                # Resize repatched image to match original size
                repatched_img = repatched_img.resize(original_img.size, Image.LANCZOS)

                # Create a new image with the height of both images
                total_height = original_img.height + repatched_img.height
                comparison_img = Image.new('RGB', (original_img.width, total_height))

                # Paste the images vertically
                comparison_img.paste(original_img, (0, 0))
                comparison_img.paste(repatched_img, (0, original_img.height))

                # Save the vertical comparison
                comparison_filename = f"{os.path.splitext(filename)[0]}_comparison.png"
                comparison_path = os.path.join(output_dir, comparison_filename)
                comparison_img.save(comparison_path)
                print(f"Saved vertical comparison: {comparison_filename}")
            else:
                print(f"Repatched image not found for {filename}")

def create_tile_vertical_comparisons(input_dir, tiles_dir, downscaled_tiles_dir, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return

    # Group tiles by their original image
    tile_groups = {}
    for filename in os.listdir(tiles_dir):
        if filename.endswith('.png') and 'tile' in filename:
            parts = filename.split('_')
            original_name = '_'.join(parts[:-3])  # Exclude 'tile', row, and column
            if original_name not in tile_groups:
                tile_groups[original_name] = []
            tile_groups[original_name].append(filename)

    for original_name, tiles in tile_groups.items():
        # Find the original image
        original_image_path = os.path.join(input_dir, f"{original_name}.png")
        if not os.path.exists(original_image_path):
            print(f"Original image not found for {original_name}")
            continue

        original_img = Image.open(original_image_path)

        for tile_name in tiles:
            # Open original tile
            original_tile_path = os.path.join(tiles_dir, tile_name)
            original_tile = Image.open(original_tile_path)

            # Open downscaled tile
            downscaled_tile_name = f"{os.path.splitext(tile_name)[0]}_downscaled.png"
            downscaled_tile_path = os.path.join(downscaled_tiles_dir, downscaled_tile_name)
            if not os.path.exists(downscaled_tile_path):
                print(f"Downscaled tile not found: {downscaled_tile_name}")
                continue
            downscaled_tile = Image.open(downscaled_tile_path)

            # Upscale the downscaled tile to match the original tile size
            downscaled_tile_upscaled = downscaled_tile.resize(original_tile.size, Image.LANCZOS)

            # Create a new image for the vertical comparison
            comparison_img = Image.new('RGB', (original_tile.width, original_tile.height * 2))

            # Paste the tiles vertically
            comparison_img.paste(original_tile, (0, 0))
            comparison_img.paste(downscaled_tile_upscaled, (0, original_tile.height))

            # Save the vertical comparison
            comparison_filename = f"{os.path.splitext(tile_name)[0]}_comparison.png"
            comparison_path = os.path.join(output_dir, comparison_filename)
            comparison_img.save(comparison_path)
            print(f"Saved tile vertical comparison: {comparison_filename}")

def create_artifact_folders():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folders = [
        '1280_16x9_vertical_comparison',
        '1280_16x9_tile_vertical_comparison',
        '1280_16x9_repatched'
    ]

    for folder in folders:
        path = os.path.join(current_dir, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created folder: {path}")

def generate_artifacts(input_dir, downscaled_dir):
    create_artifact_folders()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    comparison_dir = os.path.join(current_dir, '1280_16x9_vertical_comparison')
    tile_comparison_dir = os.path.join(current_dir, '1280_16x9_tile_vertical_comparison')
    repatched_dir = os.path.join(current_dir, '1280_16x9_repatched')

    # Step 1: Repatch downscaled tiles
    repatch_downscaled_tiles(downscaled_dir, input_dir)

    # Step 2: Create vertical comparisons for full images
    create_vertical_comparison(input_dir, repatched_dir, comparison_dir)

    # Step 3: Create vertical comparisons for individual tiles
    create_tile_vertical_comparisons(input_dir, downscaled_dir, downscaled_dir, tile_comparison_dir)


if __name__ == '__main__':
    # Set parameters here
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, '1280_16x9_test')
    output_dir = os.path.join(current_dir, '1280_16x9_cropped')
    downscaled_dir = os.path.join(current_dir, '1280_16x9_cropped_downscaled')

    # Crop tiles to 32x32
    crop_tiles(input_dir, output_dir, tile_size=64)

    # Downscale cropped tiles (if needed)
    downscale_cropped_tiles(output_dir, downscaled_dir)

    create_tile_vertical_comparisons('1280_16x9_test', output_dir, downscaled_dir, '1280_16x9_tile_vertical_comparison')

    print("Starting artifact generation process...")
    generate_artifacts(input_dir, downscaled_dir)

    print("Artifact generation completed!")