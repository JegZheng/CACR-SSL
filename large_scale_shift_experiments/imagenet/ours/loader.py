from PIL import ImageFilter
import random

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
