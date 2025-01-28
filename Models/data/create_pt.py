import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


def make_low_res_filename(high_res_filename:str):
    """
    Given a high res filename, find the corresponding low res filename.
    """
    # low_res_filename = os.path.join(low_res_dir, os.path.basename(high_res_filename))

    # Get the split. Example: os.path.splitext('path/to/br.uh.txt') -> ('path/to/br.uh', '.txt')
    split = os.path.splitext(high_res_filename)
    low_res_filename = split[0] + "_downscaled" + split[1]
    
    return low_res_filename

if __name__ == '__main__':
    high_res_dir = './challenge/challenge_64x64'
    low_res_dir = './challenge/challenge_32x32'

    high_res_list = []
    low_res_list  = []

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    print("Starting...")
    for high_res_filename in tqdm(os.listdir(high_res_dir)):
        high_res_file_path = os.path.join(high_res_dir, high_res_filename)
        high_res_image = Image.open(os.path.join(high_res_dir, high_res_filename))
        
        low_res_filename = make_low_res_filename(high_res_filename)
        low_res_file_path = os.path.join(low_res_dir, low_res_filename)
        low_res_image = Image.open(low_res_file_path)
        #print(low_res_filename)
        # if(not os.path.exists(high_res_file_path)):
        #     print(f"ERROR: High res path does not exist: {high_res_file_path}")
        #     exit()
        # if(not os.path.exists(low_res_file_path)):
        #     print(f"ERROR: Low res path does not exist: {low_res_file_path}")
        #     exit()
        high_res_tensor = transform(high_res_image)
        low_res_tesnor  = transform(low_res_image)
        
        high_res_list.append(high_res_tensor)
        low_res_list.append(low_res_tesnor)
        
        high_res_image.close()
        low_res_image.close()
        
    print("INFO [create_pt.py] All files good")
    print("INFO [create_pt.py] Stacking tensors")
    low_res_tensors = torch.stack(low_res_list)
    high_res_tensors = torch.stack(high_res_list)

    print(f"INFO [create_pt.py] High res tensor shape: {high_res_tensors.shape}")
    print(f"INFO [create_pt.py] Low res tensor shape:  {low_res_tensors.shape}")

    torch.save(low_res_tensors, "./challenge/challenge_low_res.pt")
    torch.save(high_res_tensors, "./challenge/challenge_high_res.pt")