import os
import math
import torch
import threading
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

#TODO
"""

Add composite video artifacts (ie. add filters rotate etc) so that the image processing can look for that
and fix it if there are any issues in the video feed later.
 
 """

def calculate_tiles(width: int, aspect_ratio: tuple, min_tile_size: int = 32) -> dict:
    """
    Calculate the optimal tile size and number of tiles for a given width and aspect ratio.
    This function determines the best tile size that can evenly divide the given width and height,
    which is derived from the aspect ratio. It returns a dictionary containing the optimal tile size,
    the total number of tiles, and the number of tiles horizontally and vertically.
    Args:
        width (int): The width of the area to be tiled.
        aspect_ratio (tuple): A tuple representing the aspect ratio (width, height).
        min_tile_size (int, optional): The minimum allowable tile size. Defaults to 32.
    Returns:
        dict: A dictionary containing the following keys:
            - "tile_size" (int): The optimal tile size.
            - "tile_count" (int): The total number of tiles.
            - "tiles_horizontal" (int): The number of tiles horizontally.
            - "tiles_vertical" (int): The number of tiles vertically.
    Raises:
        ValueError: If no valid tile size is found that can evenly divide the width and height.
    
    """
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

def crop_and_save_image(filenames: list, input_dir:str, output_dir: str, tile_size:int):
    """
    Crops images into smaller tiles and saves them to the specified output directory.
    
    Args:
        filenames (list): List of image filenames to be processed.
        input_dir (str): Directory where the input images are located.
        output_dir (str): Directory where the cropped image tiles will be saved.
        tile_size (int): Size of each tile (both width and height are the same).
    Returns:
        None
    Notes:
        - Only images with extensions '.png', '.jpg', and '.jpeg' are processed.
        - The tiles are saved with filenames indicating their position in the original image.
    """
    
    for filename in filenames:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            
            input_path = os.path.join(input_dir, filename)
            image = Image.open(input_path)
            width, height = image.size

            num_tiles_h = (width  + tile_size - 1) // tile_size
            num_tiles_v = (height + tile_size - 1) // tile_size

            for i in range(num_tiles_v):
                for j in range(num_tiles_h):
                    
                    # Calculate tile coordinates\
                    left = min(j * tile_size, width - tile_size)
                    top = min(i * tile_size, height - tile_size)
                    right = left + tile_size
                    bottom = top + tile_size

                    tile = image.crop((left, top, right, bottom))

                    tile_filename = f"{os.path.splitext(filename)[0]}_tile_{i}_{j}.png"
                    tile.save(os.path.join(output_dir, tile_filename))


def crop_tiles(input_dir: str, output_dir: str, tile_size: int, n_threads:int):
    """
    From images in input_dir, crop tiles of size tile_size and save them to output_dir. Run with n_threads
    
    Args:
        input_dir(str): Directory containing images to tile
        output_dir(str): Directory to save tiles. Create directory if doesn't exist
        tile_size(int): Size of tiles to crop. Square tiles.
        n_threads(int): Number of threads to process.
        
    Returns:
        None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return

    filenames = os.listdir(input_dir)
    
    chunk_size = len(filenames) // n_threads
    threads = []
    
    total_handled = 0
    for i in range(n_threads):
        start = i * chunk_size
        end = start + chunk_size if i < n_threads - 1 else len(filenames)
        chunk = filenames[start:end]

        total_handled += len(chunk)

        thread = threading.Thread(target=crop_and_save_image, args=(chunk,input_dir, output_dir, tile_size))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
        
    print(f"Total files handled: {total_handled}")

def downscale_image(filenames:list, input_dir:str, output_dir:str, with_tqdm = False):
    """
    Given a list of image filenames and their parent directory, downscale each and put into output_dir
    
    Args:
        filenames(list): List of filenames to process
        input_dir(str): Directory where filenames can be found
        output_dir(str): Directory of where to save downscaled images
        with_tqdm(bool): Set to true to print with tqdm
        
    Returns: 
        None
    """
    for filename in tqdm(filenames, disable=(not with_tqdm)):
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)

            # Create the new filename
            new_filename = f"{os.path.splitext(filename)[0]}_downscaled.png"
            output_path = os.path.join(output_dir, new_filename)

            try:
                with Image.open(input_path) as img:
                    # Calculate new dimensions
                    new_width  = img.width  // 2
                    new_height = img.height // 2

                    # Resize and save the image
                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                    resized_img.save(output_path)
                #print(f"Saved downscaled image: {new_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def downscale_images_from_dir(input_dir: str, output_dir: str, n_threads: int, factor: int = 4):
    """
    For each image in input_dir, downscale it and save it to output dir. Run this method with n_threads
    to help alleviate I/O constraints. If n_threads is 0 or 1, do not make threads

    Args:
        input_dir(str): Path to directory containing images to crop
        output_dir(str): Directory to put downscaled images. Create directory if doesn't exist
        n_threads(int): Number of threads to process input dir
        factor(int): Downscale factor. Default: 4

    Returns:
        None
    """
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Error creating output directory '{output_dir}': {e}")
        return

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory '{input_dir}' does not exist")
        return

    # Get list of files
    try:
        filenames = os.listdir(input_dir)
        if not filenames:
            print(f"ERROR: No files found in input directory '{input_dir}'")
            return
    except OSError as e:
        print(f"ERROR: Error reading input directory '{input_dir}': {e}")
        return

    print(f"INFO: Found {len(filenames)} files in '{input_dir}'")
    print(f"INFO: Downscaling images with factor {factor}...")

    if n_threads < 2:
        print(f"INFO: Running single thread")
        downscale_image(filenames, input_dir, output_dir, with_tqdm=True)
    else:
        chunk_size = len(filenames) // n_threads
        threads = []

        total_handled = 0
        for i in range(n_threads):
            start = i * chunk_size
            end = start + chunk_size if i < n_threads - 1 else len(filenames)
            chunk = filenames[start:end]

            total_handled += len(chunk)

            thread = threading.Thread(
                target=downscale_image,
                args=(chunk, input_dir, output_dir)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print(f"INFO: Total files handled: {total_handled}")

    # Copy tile info files
    for filename in os.listdir(input_dir):
        if filename.endswith('_info.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
                f_out.write(f_in.read())

#TODO: Refactor this function
def repatch_downscaled_tiles(downscaled_dir, input_dir):
    """
    
    """
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
    """
    
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png')):
            original_path = os.path.join(input_dir, filename)
            original_img = Image.open(original_path)

            # Find corresponding repatched image
            repatched_filename = f"{os.path.splitext(filename)[0]}_repatched.png"
            repatched_path = os.path.join(repatched_dir, repatched_filename)

            if os.path.exists(repatched_path):
                repatched_img = Image.open(repatched_path)

                repatched_img = repatched_img.resize(original_img.size, Image.LANCZOS)

                # Create a new image with the height of both images
                total_height = original_img.height + repatched_img.height
                comparison_img = Image.new('RGB', (original_img.width, total_height))

                # Paste the images vertically
                comparison_img.paste(original_img, (0, 0))
                comparison_img.paste(repatched_img, (0, original_img.height))

                # Save comparison
                comparison_filename = f"{os.path.splitext(filename)[0]}_comparison.png"
                comparison_path = os.path.join(output_dir, comparison_filename)
                comparison_img.save(comparison_path)
                print(f"Saved vertical comparison: {comparison_filename}")
            else:
                print(f"Repatched image not found for {filename}")

def create_tile_vertical_comparisons(input_dir, tiles_dir, downscaled_tiles_dir, output_dir):
    """
    
    """
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

# def create_artifact_folders():
#     """
    
#     """
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     folders = [
#         '1280_16x9_vertical_comparison',
#         '1280_16x9_tile_vertical_comparison',
#         '1280_16x9_repatched'
#     ]

#     for folder in folders:
#         path = os.path.join(current_dir, folder)
#         os.makedirs(path, exist_ok=True)
#         print(f"Created folder: {path}")

# def generate_artifacts(input_dir, downscaled_dir):
#     create_artifact_folders()

#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     comparison_dir = os.path.join(current_dir, '1280_16x9_vertical_comparison')
#     tile_comparison_dir = os.path.join(current_dir, '1280_16x9_tile_vertical_comparison')
#     repatched_dir = os.path.join(current_dir, '1280_16x9_repatched')

#     repatch_downscaled_tiles(downscaled_dir, input_dir)

#     create_vertical_comparison(input_dir, repatched_dir, comparison_dir)

#     create_tile_vertical_comparisons(input_dir, downscaled_dir, downscaled_dir, tile_comparison_dir)

def compute_image_variance_pil(pil_image):
    """
    Compute the variance of a PIL image.
    
    Args:
        pil_image (PIL.Image): Input image in PIL format (can be grayscale or RGB)
    
    Returns:
        variance (float): The variance of the image's pixel values
    """
    # Convert the PIL image to a NumPy array
    image = np.array(pil_image)
    
    # If the image is RGB, convert it to grayscale
    if len(image.shape) == 3:  # If the image has 3 channels (RGB)
        image = np.dot(image[..., :3], [0.299, 0.587, 0.114])  # Standard grayscale conversion

    # Normalize pixel values to the range [0, 1]
    image = image / 255.0
    
    # Compute the mean and variance
    variance = np.var(image)
    
    return variance

def create_challenge_tiles(input_dir:str, output_dir:str, info_file_path:str):
    """
    For every image in input dir, crop the 3 locations that FSRCNN has trouble upscaling into 64x64 tiles.
    Save to output_dir if the variance is high enough
    """
    try:
        os.makedirs(output_dir, exist_ok=False)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        raise e
    
    with open("./challenge/tile_variance.txt", 'w') as frame_var:
        for filename in tqdm(os.listdir(input_dir)):
            path = os.path.join(input_dir, filename)
            image = Image.open(path)
            tiles = []
            tiles.append(image.crop((84, 42, 84 + 64, 42 + 64)))
            tiles.append(image.crop((52, 95, 52 + 64, 95 + 64)))
            tiles.append(image.crop((1150, 42, 1150 + 64, 42 + 64)))

            for i in range(len(tiles)):
                tile_filename = f"{os.path.splitext(filename)[0]}_tile_{i}.png"
                tile_variance = compute_image_variance_pil(tiles[i])
                if(tile_variance > 0.01):
                    frame_var.write(f"{tile_filename:<22}: {tile_variance}\n")
                    tiles[i].save(os.path.join(output_dir, tile_filename))
    
if __name__ == '__main__':
    # Set parameters here
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, '1280_16x9_test')
    output_dir = os.path.join(current_dir, 'challenge/challenge_112x112')
    downscaled_dir = os.path.join(current_dir, 'challenge/challenge_28x28')

    # Crop tiles to 112x112
    crop_tiles(input_dir, output_dir, tile_size=112, n_threads=5)
    
    # For every image in input dir, crop a 64x64 tile and save to a new dir.
    # Each tile will be of the start coins and timer to highlight issues with FSRCNN
    # From each image, take the following image coordinates:
    # (84,42)
    # (52,95)
    # (1150,42)
    
    # Example to use image variance measurement
    # low_var_path = "C:/Users/bucht/OneDrive/Desktop/SR_MQP/Models/data/1280_16x9/frame14.png"
    # high_var_path = "C:/Users/bucht/OneDrive/Desktop/SR_MQP/Models/data/1280_16x9/frame2683.png"
    # print(compute_image_variance_pil(Image.open(low_var_path)))
    # print(compute_image_variance_pil(Image.open(high_var_path)))
    # exit()
    
    downscale_images_from_dir(output_dir, downscaled_dir, n_threads=1, factor=4)

    #create_tile_vertical_comparisons('1280_16x9_test', output_dir, downscaled_dir, '1280_16x9_tile_vertical_comparison')

    #print("Starting artifact generation process...")
    #generate_artifacts(input_dir, downscaled_dir)

    #print("Artifact generation completed!")