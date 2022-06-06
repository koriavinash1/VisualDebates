import torch.utils.data as data

from PIL import Image
import os
import numpy as np
import random
import os.path

IMG_EXTENSIONS = ['.tif','.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir, include_classes='all'):
    if include_classes == 'all':
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    else:
        classes = [d for d in os.listdir(dir) if (os.path.isdir(os.path.join(dir, d)) and d in include_classes)]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        if target not in class_to_idx.keys(): continue

        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    random.shuffle(images)
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DataGenerator(data.Dataset):
    """

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, 
                transform=None, 
                target_transform=None,
                patches = False,
                npatches = 10,
                patchsize = None,
                include_classes='all',
                loader=default_loader):
        
        if not (include_classes == 'all'):
            include_classes = include_classes.replace(' ', '')
            include_classes = [cls_ for cls_ in include_classes.split(',')]
        classes, class_to_idx = find_classes(root, include_classes)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs #[:15000]
        self.classes = classes
        self.num_class = len(classes)
        self.num_channels = 3
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.patches = patches
        self.npatches = npatches
        self.patch_size = patchsize

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        _, H, W = img.shape
        if self.patches:
            x = np.random.randint(0, W - self.patch_size, self.npatches)[..., None]
            y = np.random.randint(0, H - self.patch_size, self.npatches)[..., None]
            x = x*2./W - 1.
            y = y*2./H - 1. 
            lts = np.concatenate([y, x], axis=1)
            return img, target, lts

        return img, target, 0

    def __len__(self):
        return len(self.imgs)