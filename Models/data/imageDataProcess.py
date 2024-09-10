import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms



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
        self.image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
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

        print("naming")

        for i in range(num_tiles_v):
            for j in range(num_tiles_h):
                tile = image[:, i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size]

                tile_img = transforms.ToPILImage()(tile)

                tile_filename = f"{os.path.splitext(file_name)[0]}_tile_{i}_{j}.png"
                print("resizing")
                try:
                    tile_img.save(os.path.join(output_dir, tile_filename))
                except OSError as e:
                    print(f"Error saving tile {tile_filename}: {e}")

    print(f"Tiles saved in {output_dir}")

if __name__ == '__main__':
    # Set your parameters here
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, '1280_16x9_1000')
    output_dir = os.path.join(current_dir, '1280_16x9_1000_cropped')
    upscale_size = 2
    aspect_ratio = (16, 9)
    min_tile_size = 79

    print(f"Cropping images from {input_dir} into tiles...")
    print(f"Output directory: {output_dir}")
    print(f"Upscale size: {upscale_size}")
    print(f"Aspect ratio: {aspect_ratio}")
    print(f"Minimum tile size: {min_tile_size}")

    crop_tiles(input_dir, output_dir, upscale_size, aspect_ratio, min_tile_size)

    print("Cropping completed!")





def downscale_cropped_tiles():
    # function call to crop_tiles to downscale them one at a time with a for loop.
    # use an array through the file and save the new images

    path = "1280_16x9_1000_cropped"
    output_path = "1280_16x9_cropped_downscaled"
    dirs = os.listdir(path)
    output = os.listdir(output_path)

    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            imResize = im.resize((40, 40), Image.ANTIALIAS)
            imResize.save(f + ' downscaled.png', 'PNG', quality=90)


    downscale_cropped_tiles()


#def png_output():
    # function call to downscale_cropped_tiles and crop_tiles to create atrifact images and
    # save the different sizes with image artifacts (pre downscaled image with downscales image)