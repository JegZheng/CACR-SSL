from PIL import ImageFilter, Image, ImageFile
import random
from torch.utils.data import Dataset
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Transform for K = 1
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

# Transform for K > 1
class MultiCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, Ny):
        self.base_transform = base_transform
        self.trans = [self.base_transform for _ in range(Ny)]

    def __call__(self, x):
        return list(map(lambda trans: trans(x), self.trans))


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class webvision_dataset(Dataset): 
    def __init__(self, args, transform, num_class=1000, is_train=True): 
        self.root = args.data
        self.transform = transform

        self.train_imgs = []
        self.train_labels = {}    
             
        with open(os.path.join(self.root, 'info/train_filelist_google.txt')) as f:
            lines=f.readlines()    
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.train_imgs.append(img)
                    self.train_labels[img]=target            
        
        with open(os.path.join(self.root, 'info/train_filelist_flickr.txt')) as f:
            lines=f.readlines()    
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.train_imgs.append(img)
                    self.train_labels[img]=target            

    def __getitem__(self, index):

        img_path = self.train_imgs[index]
        target = self.train_labels[img_path]
        file_path = os.path.join(self.root, img_path)

        image = Image.open(file_path).convert('RGB')   
        img = self.transform(image)        

        return img, target
                

    def __len__(self):
        return len(self.train_imgs)