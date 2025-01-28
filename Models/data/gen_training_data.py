import os
import math
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
import threading
from tqdm import tqdm
import torch
import torchvision.transforms as T


class CompositeVideoAugmenter:
    def __init__(self, p=0.5):
        self.p = p
        self.transforms = T.Compose([
            T.RandomApply([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ], p=p),
            T.RandomApply([
                T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.3))
            ], p=0.15),
            T.RandomAdjustSharpness(sharpness_factor=0.5, p=p),
            T.RandomApply([
                T.RandomAffine(degrees=1, translate=(0.02, 0.02), scale=(0.98, 1.02))
            ], p=p)
        ])

    def __call__(self, img):
        return self.transforms(img)


def get_video_parameters():
    """Generate fixed parameters for the Wii-like composite video effects"""
    return {
        'color_blur': 2.0,
        'rf_noise_intensity': 0.0002,
        'scanline_intensity': 0.003,
        'blur_amount': 0.93
    }


def apply_color_bleeding(img_array, intensity):
    """Apply horizontal color bleeding effect"""
    y = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
    u = -0.147 * img_array[:, :, 0] - 0.289 * img_array[:, :, 1] + 0.436 * img_array[:, :, 2]
    v = 0.615 * img_array[:, :, 0] - 0.515 * img_array[:, :, 1] - 0.100 * img_array[:, :, 2]

    u = gaussian_filter(u, sigma=[0, intensity])
    v = gaussian_filter(v, sigma=[0, intensity])

    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u

    return np.stack([r, g, b], axis=2)


def apply_composite_artifacts(image):
    """Apply composite video artifacts to simulate Wii video output"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.filter(ImageFilter.GaussianBlur(radius=get_video_parameters()['blur_amount']))
    img_array = np.array(image).astype(np.float32) / 255.0
    params = get_video_parameters()

    img_array = apply_color_bleeding(img_array, params['color_blur'])

    height, width = img_array.shape[:2]
    scanlines = np.ones_like(img_array)
    scanlines[::2, :, :] *= (1 - params['scanline_intensity'])
    img_array *= scanlines

    noise = np.random.normal(0, params['rf_noise_intensity'], img_array.shape)
    img_array += noise

    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def apply_composite_artifacts_augmented(image, augmenter=None):
    """Enhanced version of apply_composite_artifacts with additional augmentations"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply base composite artifacts
    image = apply_composite_artifacts(image)

    # Convert to torch tensor for augmentations
    if augmenter is not None:
        tensor_img = T.ToTensor()(image)
        augmented_tensor = augmenter(tensor_img)
        image = T.ToPILImage()(augmented_tensor)

    return image


def should_process_file(filename: str) -> bool:
    """Check if the file should be processed based on its extension"""
    return (not filename.startswith('.') and
            not filename.startswith('._') and
            filename.lower().endswith(('.png', '.jpg', '.jpeg')))


def crop_and_save_image(filenames: list, input_dir: str, output_dir: str, tile_size: int):
    """Crops images into smaller tiles"""
    for filename in filenames:
        if should_process_file(filename):
            input_path = os.path.join(input_dir, filename)
            image = Image.open(input_path)

            width, height = image.size
            num_tiles_h = (width + tile_size - 1) // tile_size
            num_tiles_v = (height + tile_size - 1) // tile_size

            for i in range(num_tiles_v):
                for j in range(num_tiles_h):
                    left = min(j * tile_size, width - tile_size)
                    top = min(i * tile_size, height - tile_size)
                    right = left + tile_size
                    bottom = top + tile_size

                    tile = image.crop((left, top, right, bottom))
                    tile_filename = f"{os.path.splitext(filename)[0]}_tile_{i}_{j}.png"
                    tile.save(os.path.join(output_dir, tile_filename))


def crop_tiles(input_dir: str, output_dir: str, tile_size: int, n_threads: int):
    """Multi-threaded tile cropping"""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        return

    filenames = os.listdir(input_dir)
    chunk_size = len(filenames) // n_threads
    threads = []

    for i in range(n_threads):
        start = i * chunk_size
        end = start + chunk_size if i < n_threads - 1 else len(filenames)
        chunk = filenames[start:end]

        thread = threading.Thread(target=crop_and_save_image,
                                  args=(chunk, input_dir, output_dir, tile_size))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def downscale_image_augmented(filenames: list, input_dir: str, output_dir: str, factor: int, augmenter=None,
                              with_tqdm=False):
    """Enhanced version of downscale_image with augmentations"""
    for filename in tqdm(filenames, disable=(not with_tqdm)):
        if should_process_file(filename):
            input_path = os.path.join(input_dir, filename)

            for aug_idx in range(3):
                new_filename = f"{os.path.splitext(filename)[0]}_aug{aug_idx}.png"
                output_path = os.path.join(output_dir, new_filename)

                try:
                    with Image.open(input_path) as img:
                        new_width = img.width // factor
                        new_height = img.height // factor
                        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                        processed_img = apply_composite_artifacts_augmented(resized_img, augmenter)
                        processed_img.save(output_path)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


def downscale_images_from_dir_augmented(input_dir: str, output_dir: str, n_threads: int, factor: int = 4):
    """Multi-threaded image downscaling with composite artifacts and augmentations"""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Error creating output directory: {e}")
        return

    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory does not exist")
        return

    augmenter = CompositeVideoAugmenter(p=0.5)
    filenames = os.listdir(input_dir)

    if not filenames:
        print(f"ERROR: No files found in input directory")
        return

    if n_threads < 2:
        downscale_image_augmented(filenames, input_dir, output_dir, factor, augmenter, with_tqdm=True)
    else:
        chunk_size = len(filenames) // n_threads
        threads = []

        for i in range(n_threads):
            start = i * chunk_size
            end = start + chunk_size if i < n_threads - 1 else len(filenames)
            chunk = filenames[start:end]

            thread = threading.Thread(
                target=downscale_image_augmented,
                args=(chunk, input_dir, output_dir, factor, augmenter)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


if __name__ == '__main__':
    # Set up directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, './data/1280_16x9_1')
    output_dir = os.path.join(current_dir, './data/1280_16x9_tiles_hi_res4x')
    downscaled_dir = os.path.join(current_dir, './data/1280_16x9_tiles_downscaled4x_augmented')

    tile_size = 112
    n_threads = 5
    downscale_factor = 4

    print("Cropping images into tiles...")
    crop_tiles(input_dir, output_dir, tile_size=tile_size, n_threads=n_threads)

    print("Downscaling tiles and applying effects with augmentations...")
    downscale_images_from_dir_augmented(
        output_dir,
        downscaled_dir,
        n_threads=n_threads,
        factor=downscale_factor
    )

    print("Processing complete!")