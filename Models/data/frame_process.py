import os
import sys
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    
    srcdir = './1280_16x9'
    newdir = './640_16x9'
    os.mkdir(newdir)
    #print(newpath)
        
    for file in tqdm(os.listdir(srcdir)):
        
        filename = os.fsdecode(file)
        fullPathToFile = os.path.join(srcdir, filename)
        
        try:
            im = Image.open(fullPathToFile)
            
            #im = im.resize((3840, 2160), Image.Resampling.LANCZOS)
            #im = im.resize((3840, 2160), Image.Resampling.NEAREST)
            #im = im.resize((1280, 720), Image.Resampling.NEAREST)
            im = im.resize((640, 360), Image.Resampling.NEAREST)
            
            newFileName = filename[:-4] + '.png'
            newFilePath = os.path.join(newdir, newFileName)
            
            im.save(newFilePath, optimize=False)
        
        except IOError:
            print(f"IOError on file \'{filename}\'")
        
        
        
        