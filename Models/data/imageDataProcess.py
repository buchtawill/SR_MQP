import os
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


def crop_tiles(input_dir: str, output_dir: str, upscale_size: int, aspect_ratio: tuple, min_tile_size: int = 32):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return

    dataset = ImageDataset(input_dir)

    for idx in range(len(dataset)):
        image, file_name = dataset[idx]

        # Extract image size from filename or use actual image size
        try:
            image_width = int(file_name.split('_')[0])
        except (IndexError, ValueError):
            image_width = image.shape[2]  # Assuming image is already a tensor

        tile_info = calculate_tiles(image_width, aspect_ratio, min_tile_size)
        tile_size = tile_info['tile_size']

        num_tiles_h = image.shape[2] // tile_size
        num_tiles_v = image.shape[1] // tile_size

        print(f"Processing {file_name}")

        for i in range(num_tiles_v):
            for j in range(num_tiles_h):
                tile = image[:, i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size]

                tile_img = transforms.ToPILImage()(tile)

                tile_filename = f"{os.path.splitext(file_name)[0]}_tile_{i}_{j}.png"
                try:
                    tile_img.save(os.path.join(output_dir, tile_filename))
                except OSError as e:
                    print(f"Error saving tile {tile_filename}: {e}")

    print(f"Tiles saved in {output_dir}")


def downscale_cropped_tiles(input_dir: str, output_dir: str):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png')):
            input_path = os.path.join(input_dir, filename)

            # Create the new filename
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_downscaled{ext}"
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

def repatch_downscaled_tiles(downscaled_dir, input_dir):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repatched_dir = os.path.join(current_dir, '1280_16x9_repatched')
    os.makedirs(repatched_dir, exist_ok=True)

    # Group tiles by their original image
    tile_groups = {}
    for filename in os.listdir(downscaled_dir):
        if filename.endswith('_downscaled.png'):
            parts = filename.split('_')
            if len(parts) < 4:
                print(f"Skipping file with unexpected format: {filename}")
                continue
            original_name = '_'.join(parts[:-4])  # Exclude 'tile', row, column, and 'downscaled' info
            if not original_name:
                print(f"Skipping file with empty original name: {filename}")
                continue
            if original_name not in tile_groups:
                tile_groups[original_name] = []
            tile_groups[original_name].append(filename)

    # Repatch each group
    for original_name, tiles in tile_groups.items():
        if not tiles:
            print(f"No tiles found for {original_name}")
            continue

        # Sort tiles by row and column
        sorted_tiles = sorted(tiles, key=lambda x: (
            int(x.split('_')[-3]),  # Row
            int(x.split('_')[-2])   # Column
        ))

        # Open first tile to get dimensions
        first_tile = Image.open(os.path.join(downscaled_dir, sorted_tiles[0]))
        tile_width, tile_height = first_tile.size

        # Determine the number of tiles in each dimension
        num_tiles_vertical = max(int(x.split('_')[-3]) for x in sorted_tiles) + 1
        num_tiles_horizontal = max(int(x.split('_')[-2]) for x in sorted_tiles) + 1

        # Calculate the dimensions of the repatched image
        repatched_width = tile_width * num_tiles_horizontal
        repatched_height = tile_height * num_tiles_vertical

        # Create a new image for the repatched result
        repatched_img = Image.new('RGB', (repatched_width, repatched_height))

        # Place each tile in the correct position
        for tile_name in sorted_tiles:
            tile = Image.open(os.path.join(downscaled_dir, tile_name))
            row = int(tile_name.split('_')[-3])
            col = int(tile_name.split('_')[-2])
            repatched_img.paste(tile, (col * tile_width, row * tile_height))

        # Get the dimensions of the original image
        original_image_path = os.path.join(input_dir, f"{original_name}.png")
        if not os.path.exists(original_image_path):
            print(f"Original image not found for {original_name}")
            continue

        original_img = Image.open(original_image_path)
        original_width, original_height = original_img.size

        # Upscale the repatched image to match the original dimensions
        repatched_img_upscaled = repatched_img.resize((original_width, original_height), Image.NEAREST)

        # Save the upscaled repatched image
        repatched_img_upscaled.save(os.path.join(repatched_dir, f"{original_name}_repatched.png"))
        print(f"Saved upscaled repatched image: {original_name}_repatched.png")

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

                # Create a new image with the height of both images
                total_height = original_img.height + repatched_img.height
                max_width = max(original_img.width, repatched_img.width)
                comparison_img = Image.new('RGB', (max_width, total_height))

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
        if filename.endswith('_downscaled.png'):
            parts = filename.split('_')
            if len(parts) < 4:
                print(f"Skipping file with unexpected format: {filename}")
                continue
            original_name = '_'.join(parts[:-4])  # Exclude 'tile', row, column, and 'downscaled' info
            if not original_name:
                print(f"Skipping file with empty original name: {filename}")
                continue
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
            # Open downscaled tile
            downscaled_tile_path = os.path.join(downscaled_tiles_dir, tile_name)
            downscaled_tile = Image.open(downscaled_tile_path)

            # Calculate the position of the tile in the original image
            parts = tile_name.split('_')
            row = int(parts[-3])
            col = int(parts[-2])
            tile_size = downscaled_tile.width * 2  # Assuming downscaled tiles are half the size

            # Crop the corresponding area from the original image
            original_tile = original_img.crop((col * tile_size, row * tile_size,
                                               (col + 1) * tile_size, (row + 1) * tile_size))

            # Upscale the downscaled tile to match the original tile size
            downscaled_tile_upscaled = downscaled_tile.resize(original_tile.size, Image.NEAREST)

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
    downscaled_dir = os.path.join(current_dir, '1280_16x9_cropped_downscaled')
    upscale_size = 2
    aspect_ratio = (16, 9)
    min_tile_size = 79

    print("Starting artifact generation process...")
    generate_artifacts(input_dir, downscaled_dir)
    print("Artifact generation completed!")