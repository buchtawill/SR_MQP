import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

"""
This dataset assumes your directory is structured like:
highres/
| frame0.png
lowres/
| frame0_downscaled.png

"""


def make_low_res_filename(high_res_filename:str):
    """
    Given a high res filename, find the corresponding low res filename.
    """
    # low_res_filename = os.path.join(low_res_dir, os.path.basename(high_res_filename))

    # Get the split. Example: os.path.splitext('path/to/br.uh.txt') -> ('path/to/br.uh', '.txt')
    split = os.path.splitext(high_res_filename)
    low_res_filename = split[0] + "_downscaled" + split[1]
    
    return low_res_filename

class SR_image_dataset(Dataset):
    """
    Load images and convert them to tensors.
    """
    
    def __init__(self, lowres_path:str, highres_path:str, transform=transforms.ToTensor):
        """
        Find all of the image names in the path specified by highres_path, find out what the corresponding lowres image name is,
        then load those as well. Put all of the filenames into two lists. Raise exception if error.
        """
        self.lowres_dir  = lowres_path
        self.highres_dir = highres_path
        self.transform = transform
        self.high_res_paths = None
        self.low_res_paths  = None
        
        if(not os.path.isdir(lowres_path)):
            raise Exception(f"ERROR [SR_image_dataset::__init__()] Cannot find dir at '{lowres_path}'")
        
        if(not os.path.isdir(highres_path)):
            raise Exception(f"ERROR [SR_image_dataset::__init__()] Cannot find dir at '{highres_path}'")
        
        # Get filenames of high res photos
        #self.high_res_paths = np.array([os.path.join(self.highres_dir, file) for file in os.listdir(self.highres_dir)])
        self.high_res_paths = os.listdir(self.highres_dir)
        self.low_res_paths  = os.listdir(self.highres_dir) #just to initialize
        
        # Get filenames of low res photos
        for i in range(len(self.high_res_paths)):
            high_res_name = self.high_res_paths[i]
            self.low_res_paths[i] = make_low_res_filename(high_res_name)
            
            # Modify lists to be the full path to the image
            full_highres_path = os.path.join(self.highres_dir, self.high_res_paths[i])
            full_lowres_path  = os.path.join(self.lowres_dir, self.low_res_paths[i])
            
            self.high_res_paths[i] = full_highres_path
            self.low_res_paths[i]  = full_lowres_path
            
            if(not os.path.exists(self.low_res_paths[i])):
                raise Exception(f"ERROR [SR_image_dataset::__init__()] Cannot find image '{self.low_res_paths[i]}'")
           
            if(not os.path.exists(self.high_res_paths[i])):
                raise Exception(f"ERROR [SR_image_dataset::__init__()] Cannot find image '{self.high_res_paths[i]}'")
        
        # highres_image = Image.open(full_highres_path)
        # lowres_image  = Image.open(full_lowres_path)
        # highres_image.show()
        # lowres_image.show()
        # exit()
        
    def __len__(self):
        return len(self.high_res_paths)
        
    def __getitem__(self, idx):
        """
        Return a tuple of the next image pair (low_res_image, high_res_image)
        """
        
        highres_path = self.high_res_paths[idx]
        lowres_path  = self.low_res_paths[idx]
        
        highres_image = Image.open(highres_path)
        lowres_image  = Image.open(lowres_path)
        
        if(self.transform):
            highres_image = self.transform(highres_image)
            lowres_image  = self.transform(lowres_image)
            
        return lowres_image, highres_image