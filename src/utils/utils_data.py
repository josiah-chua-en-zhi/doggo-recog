"""
Utils for data loader and feeding data into models
"""
"""
Pytorch trianing functions and classes to exceute training experiments
"""
import os
import random
import math
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils.utils_envvar import EnvVar

env_vars = EnvVar()
random.seed(env_vars.random_state)

class DoggoDataset(Dataset):
    """
    Dog dataset.
    Takes in the pd dataframe in __init__ but leave the reading of images to __getitem__. 
    This is memory efficient because all the images are not stored in the memory at once but read as required.
    """

    def __init__(self, dataset_df, filename_col_name, label_col_name, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_df = dataset_df
        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name
        self.transform = transform

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        """
        Gets dictionary containing an image filepath and landmarks for each respective key
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_filepath = self.dataset_df.iloc[idx][self.filename_col_name]
        label = self.dataset_df.iloc[idx][self.label_col_name]

        try:
            image = np.load(image_filepath)
        except Exception as _e:
            print(_e)
            raise Exception(image_filepath + " cannot be read")
 
        sample = {self.filename_col_name: image, self.label_col_name: label}

        if self.transform:

            sample = self.transform(sample)

        return sample
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, filename_col_name, label_col_name, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]

        h, w = image.shape[1:]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transforms.functional.resize(img = image, size = (new_h, new_w), antialias=True)

        return {self.filename_col_name: image, self.label_col_name: label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample.
    By picking a random point from 0 to difference between original size and cropped sized

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, filename_col_name, label_col_name, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]

        h, w = image.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = transforms.functional.crop(img = image, 
                                           top = top, 
                                           left = left, 
                                           height = new_h, 
                                           width = new_w)

        return {self.filename_col_name: image, self.label_col_name: label}

class RandomGreyscale(object):
    """
    RNG to see if a picture is turned into black and white based on a given probability
    Args:
        p (float <= 1): Probability of a picture being turned greyscale (optional, default 0.025)
        ch_num (int {1,3}): Number of channels to output to, either 1 or 3
    """

    def __init__(self, filename_col_name, label_col_name, ch_num, p = 0.025):
        assert isinstance(p, float) and p <= 1
        assert isinstance(ch_num, int) and ch_num in (1,3)
        self.p = p
        self.ch_num = ch_num
        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]

        grey_or_naw = random.random()

        if grey_or_naw < self.p:
            image = transforms.functional.rgb_to_grayscale(\
                img = image, num_output_channels = self.ch_num)
            
        return {self.filename_col_name: image, self.label_col_name: label}

class RandomPersepctive(object):
    """
    RNG to see if a picture has a random warped perspective based on a given probability
    Args:
        p (float <= 1): Probability of a picture being turned greyscale (optional, default 0.025)
        max_padding_ratio (float <= 0.9): Maximum percentage of transformed image that is zero 
        padded (optional, default 0.5)
    """

    def __init__(self, filename_col_name, label_col_name, p = 0.025, max_padding_ratio = 0.5):
        assert isinstance(p, float) and p <= 1
        assert isinstance(max_padding_ratio, float) and max_padding_ratio <= 0.9
        self.p = p
        self.max_padding_ratio = max_padding_ratio

        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]

        perspective_or_naw = random.random()

        if perspective_or_naw < self.p:
            h = image.shape[1]
            w = image.shape[2]
            ratio = math.sqrt(1 - self.max_padding_ratio)
            t_max = int(h/2 - h/2 * ratio)
            b_max = int(h/2 + h/2 * ratio)
            l_max = int(w/2 - w/2 * ratio)
            r_max = int(w/2 + w/2 * ratio)

            t_l_h, t_l_w = random.randint(0,t_max) , random.randint(0,l_max)
            t_r_h, t_r_w = random.randint(0,t_max) , random.randint(r_max,w)
            b_l_h, b_l_w = random.randint(b_max,h) , random.randint(0,l_max)
            b_r_h, b_r_w = random.randint(b_max,h) , random.randint(r_max,w)
            #print(f"({t_l_h, t_l_w }), ({t_r_h, t_r_w}), ({b_l_h, b_l_w}), ({b_r_h, b_r_w})")

            image = transforms.functional.perspective(img = image, 
                                                      startpoints = [[0,0], [0,w], [h,0], [h,w]],
                                                      endpoints = [[t_l_h, t_l_w], [t_r_h, t_r_w], \
                                                                   [b_l_h, b_l_w], [b_r_h, b_r_w]])
            
        return {self.filename_col_name: image, self.label_col_name: label}

class RandomRotate(object):
    """
    RNG to see if a picture is turned into black and white based on a given probability.
    Rotation is 360 degrees
    Args:
        p (float <= 1): Probability of a picture being rotated (optional, default 0.025)
        rotate_range(array (a,b) 0<={a,b}<=360): range of angle of rotation 
        min and max in degrees (optional, default (0,360))
    """

    def __init__(self, filename_col_name, label_col_name, p = 0.025, rotate_range = (0,360)):
        assert isinstance(p, float) and p <= 1
        assert isinstance(rotate_range, (list, tuple))
        assert 0 <= rotate_range[0] <= rotate_range[1] <= 360
        self.p = p
        self.rotate_range = rotate_range

        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]
        rotate_or_naw = random.random()

        if rotate_or_naw < self.p:
            min_ = self.rotate_range[0]
            max_ = self.rotate_range[1]
            angle = random.random() * (max_ - min_) + min_
            image = transforms.functional.rotate(img = image, angle = angle)

        return {self.filename_col_name: image, self.label_col_name: label}

class RandomFlip(object):
    """
    RNG to see if a picture is flipped vertically and or horizontaly based on a given probability.
    Has a chance to be both flipped vertially and horizontally
    Args:
        p (float <= 1): Probability of a picture being flipped (optional, default 0.025)
    """

    def __init__(self, filename_col_name, label_col_name, p = 0.025):
        assert isinstance(p, float) and p <= 1
        self.p = p

        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]

        flip_or_naw_v = random.random()
        flip_or_naw_h = random.random()

        if flip_or_naw_v < self.p:
            image = transforms.functional.vflip(img = image)
        if flip_or_naw_h < self.p:
            image = transforms.functional.hflip(img = image)

        return {self.filename_col_name: image, self.label_col_name: label}

class RandomBrighten(object):
    """
    RNG to see if a picture is brightened/dimmed based on a given probability.
    Args:
        p (float <= 1): Probability of a picture being flipped (optional, default 0.025)
        min_brightness (float <= 0): Minimum factor of brightness (optional, default 0.25)
        max_brightness (float <= 0): Maximum factor of brightness (optional, default 4)
    """

    def __init__(self, filename_col_name, label_col_name, p = 0.025, min_brightness = 0.25, max_brightness = 4.0):
        assert isinstance(p, float) and p <= 1
        assert isinstance(min_brightness, (int, float))
        assert isinstance(max_brightness, (int, float)) 
        assert 0 <= min_brightness <= max_brightness
        self.p = p
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]

        brighten_or_naw = random.random()

        if brighten_or_naw < self.p:
            brightness_factor = random.random() * (self.max_brightness - self.min_brightness) \
                + self.min_brightness
            image = transforms.functional.adjust_brightness(img = image, 
                                                            brightness_factor = brightness_factor)

        return {self.filename_col_name: image, self.label_col_name: label}


class RandomGaussianBlur(object):
    """
    RNG to see if a picture is blurred based on a given probability.
    Args:
        p (float <= 1): Probability of a picture being flipped (optional, default 0.025)
        kernel_height_range (list/tuple (min,max) {min,max}: int): Kernel height range (optional, default (3,5))
        kernel_width_range (list/tuple (min,max) {min,max}: int): Kernel width range (optional, default (3,5))
    """

    def __init__(self, filename_col_name, label_col_name, p = 0.025, kernel_height_range = (3,5), kernel_width_range = (3,5)):
        assert isinstance(p, float) and p <= 1
        assert isinstance(kernel_height_range, (list, tuple)) and isinstance(kernel_width_range, (list, tuple))
        assert len(kernel_height_range) == 2 and len(kernel_width_range) == 2
        assert all(isinstance(item, int) for item in kernel_height_range)
        assert all(isinstance(item, int) for item in kernel_width_range)
        assert 0 <= kernel_height_range[0] <= kernel_height_range[1]
        assert 0 <= kernel_width_range[0] <= kernel_width_range[1]
        self.p = p
        self.kernel_height_range = kernel_height_range
        self.kernel_width_range = kernel_width_range

        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]
        blur_or_naw = random.random()

        if blur_or_naw < self.p:
            h = random.randint(self.kernel_height_range[0], self.kernel_height_range[1])
            w = random.randint(self.kernel_width_range[0], self.kernel_width_range[1])
            if h%2 == 0:
                h +=1
            if w%2 == 0:
                w +=1

            kernel_size = (h,w)
            image = transforms.functional.gaussian_blur(img = image, kernel_size = kernel_size)

        return {self.filename_col_name: image, self.label_col_name: label}
        
class RandomContrast(object):
    """
    RNG to see if a picture is contrasted based on a given probability.
    Args:
        p (float <= 1): Probability of a picture being flipped (optional, default 0.025)
        min_contrast (float <= 0): Minimum factor of contrast (optional, default 0.5)
        max_contrast (float <= 0): Maximum factor of contrast (optional, default 2)
    """

    def __init__(self, filename_col_name, label_col_name, p = 0.025, min_contrast = 0.5, max_contrast = 2.0):
        assert isinstance(p, float) and p <= 1
        assert isinstance(min_contrast, (int, float))
        assert isinstance(max_contrast, (int, float)) 
        assert 0 <= min_contrast <= max_contrast
        self.p = p
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]
        contrast_or_naw = random.random()

        if contrast_or_naw < self.p:
            contrast_factor = random.random() * (self.max_contrast - self.min_contrast) + self.min_contrast
            image = transforms.functional.adjust_contrast(img = image, contrast_factor = contrast_factor)

        return {self.filename_col_name: image, self.label_col_name: label}
    
class ScalePixelValues(object):
    """Scale pixel value from 0-255 to 0-1."""

    def __init__(self, filename_col_name, label_col_name):
        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]
        image = image/255

        return {self.filename_col_name: image, self.label_col_name: label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, filename_col_name, label_col_name):

        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # make the array writable
        assert isinstance(image, np.ndarray), "Input for preprocessor has to be numpy array"
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).type(torch.int64)

        return {self.filename_col_name: image, self.label_col_name: label}

class Normalize(object):
    """Normalize image"""

    def __init__(self, filename_col_name, label_col_name):

        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):

        image, label = sample[self.filename_col_name], sample[self.label_col_name]

        image = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        return {self.filename_col_name: image, self.label_col_name: label}


class Format(object):

    """Convert into (img, label) format."""

    def __init__(self, filename_col_name, label_col_name):
        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):
        image, label = sample[self.filename_col_name], sample[self.label_col_name]
        return (image,label)
    
class PredictionFormat(object):

    """Convert into (img) format."""

    def __init__(self, filename_col_name, label_col_name):
        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

    def __call__(self, sample):
        image = sample[self.filename_col_name]
        # Convert to (1,3,224,224) shape
        return (image[None, :, :, :])
    
class DataLoaderConstructors():
    def __init__(
            self,
            train_dataset : pd.DataFrame = pd.DataFrame(),
            val_dataset : pd.DataFrame = pd.DataFrame(),
            test_dataset: pd.DataFrame = pd.DataFrame(),
            filename_col_name: str = None,
            label_col_name: str = None):
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.filename_col_name = filename_col_name
        self.label_col_name = label_col_name

        self.train_transform = None
        self.test_val_transform = None
        self.prediction_transformer = None

    def initalze_prediction_transformer(self,input_size: int = 224):
        
        self.prediction_transformer = transforms.Compose([
            ToTensor(filename_col_name = self.filename_col_name,
                     label_col_name = self.label_col_name),
            Rescale(filename_col_name = self.filename_col_name,
                    label_col_name = self.label_col_name, 
                    output_size = input_size),
            ScalePixelValues(filename_col_name = self.filename_col_name,
                             label_col_name = self.label_col_name),
            Normalize(filename_col_name = self.filename_col_name,
                      label_col_name = self.label_col_name),
            PredictionFormat(filename_col_name = self.filename_col_name,
                   label_col_name = self.label_col_name)])
        return self.prediction_transformer

    def initalize_data_transformers(
            self,
            input_size: int = 224, 
            rescale_add: int = 32,
            random_contrast_p: float = 0.05,
            random_contrast_min_contrast: float = 0.5,
            random_contrast_max_contrast: float = 2,
            random_grey_scale_p: float = 0.05,
            random_greyscale_ch_num: int = 3,
            random_persepctive_p: float = 0.05,
            random_persepctive_max_padding_ratio: float = 0.7,
            random_rorate_p: float = 0.05,
            random_rorate_rotate_range: tuple = (0, 360),
            random_flip_p: float = 0.05,
            random_brighten_p: float = 0.05,
            random_brighten_min_brightness: float = 0.33,
            random_brighten_max_brightness: float = 3,
            random_gaussian_blur_p: float = 0.05,
            random_gaussian_blur_kernel_height_range: tuple = (7, 12),
            random_gaussian_blur_kernel_width_range: tuple = (7, 12)):
        
        assert isinstance(input_size, int), "input_size must be an integer"
        assert isinstance(rescale_add, int), "rescale_add must be an integer"
        assert isinstance(random_contrast_p, float), "random_contrast_p must be a float"
        assert isinstance(random_contrast_min_contrast, float) or \
            isinstance(random_contrast_min_contrast, int), "random_contrast_min_contrast must be a float or int"
        assert isinstance(random_contrast_max_contrast, float) or\
            isinstance(random_contrast_max_contrast, int), "random_contrast_max_contrast must be a float or int"
        assert isinstance(random_grey_scale_p, float), "random_grey_scale_p must be a float"
        assert isinstance(random_greyscale_ch_num, int), "random_greyscale_ch_num must be an integer"
        assert isinstance(random_persepctive_p, float), "random_persepctive_p must be a float"
        assert isinstance(random_persepctive_max_padding_ratio, float), "random_persepctive_max_padding_ratio must be a float"
        assert isinstance(random_rorate_p, float), "random_rorate_p must be a float"
        assert isinstance(random_rorate_rotate_range, tuple) or \
            isinstance(random_rorate_rotate_range, list), "random_rorate_rotate_range must be a tuple or list"
        assert isinstance(random_flip_p, float), "random_flip_p must be a float"
        assert isinstance(random_brighten_p, float), "random_brighten_p must be a float"
        assert isinstance(random_brighten_min_brightness, float) or \
            isinstance(random_brighten_min_brightness, int), "random_brighten_min_brightness must be a float or int"
        assert isinstance(random_brighten_max_brightness, float) or \
            isinstance(random_brighten_max_brightness, int), "random_brighten_max_brightness must be a float or int"
        assert isinstance(random_gaussian_blur_p, float), "random_gaussian_blur_p must be a float"
        assert isinstance(random_gaussian_blur_kernel_height_range, tuple) or \
            isinstance(random_gaussian_blur_kernel_height_range, list), "random_gaussian_blur_kernel_height_range must be a tuple or list"
        assert isinstance(random_gaussian_blur_kernel_width_range, tuple) or \
            isinstance(random_gaussian_blur_kernel_width_range, list), "random_gaussian_blur_kernel_width_range must be a tuple or list"
        
        self.train_transform = transforms.Compose([
            ToTensor(filename_col_name = self.filename_col_name,
                     label_col_name = self.label_col_name),
            Rescale(filename_col_name = self.filename_col_name,
                    label_col_name = self.label_col_name, 
                    output_size = input_size + rescale_add),
            RandomCrop(filename_col_name = self.filename_col_name,
                       label_col_name = self.label_col_name,
                       output_size = input_size),
            RandomContrast(filename_col_name = self.filename_col_name,
                           label_col_name = self.label_col_name,
                           p = random_contrast_p,
                           min_contrast = random_contrast_min_contrast,
                           max_contrast = random_contrast_max_contrast),
            RandomGreyscale(filename_col_name = self.filename_col_name,
                            label_col_name = self.label_col_name,
                            p = random_grey_scale_p,
                            ch_num = random_greyscale_ch_num),
            RandomPersepctive(filename_col_name = self.filename_col_name,
                              label_col_name = self.label_col_name,
                              p = random_persepctive_p,
                              max_padding_ratio = random_persepctive_max_padding_ratio),
            RandomRotate(filename_col_name = self.filename_col_name,
                         label_col_name = self.label_col_name,
                         p = random_rorate_p,
                         rotate_range = random_rorate_rotate_range),
            RandomFlip(filename_col_name = self.filename_col_name,
                       label_col_name = self.label_col_name,
                       p = random_flip_p),
            RandomBrighten(filename_col_name = self.filename_col_name,
                           label_col_name = self.label_col_name,
                           p = random_brighten_p,
                           min_brightness = random_brighten_min_brightness,
                           max_brightness = random_brighten_max_brightness),
            RandomGaussianBlur(filename_col_name = self.filename_col_name,
                               label_col_name = self.label_col_name,
                               p = random_gaussian_blur_p,
                               kernel_height_range = random_gaussian_blur_kernel_height_range,
                               kernel_width_range = random_gaussian_blur_kernel_width_range),
            ScalePixelValues(filename_col_name = self.filename_col_name,
                             label_col_name = self.label_col_name),
            Normalize(filename_col_name = self.filename_col_name,
                      label_col_name = self.label_col_name),
            Format(filename_col_name = self.filename_col_name,
                   label_col_name = self.label_col_name)])

        self.test_val_transform = transforms.Compose([
            ToTensor(filename_col_name = self.filename_col_name,
                     label_col_name = self.label_col_name),
            Rescale(filename_col_name = self.filename_col_name,
                    label_col_name = self.label_col_name, 
                    output_size = input_size + rescale_add),
            RandomCrop(filename_col_name = self.filename_col_name,
                       label_col_name = self.label_col_name,
                       output_size = input_size),
            ScalePixelValues(filename_col_name = self.filename_col_name,
                             label_col_name = self.label_col_name),
            Normalize(filename_col_name = self.filename_col_name,
                      label_col_name = self.label_col_name),
            Format(filename_col_name = self.filename_col_name,
                   label_col_name = self.label_col_name)])
        
    def initalize_dataloaders(self, batch_size: int = 128):
        
        train_dataset = DoggoDataset(
            dataset_df = self.train_dataset,
            filename_col_name = self.filename_col_name, 
            label_col_name = self.label_col_name,
            transform = self.train_transform
        )

        val_dataset = DoggoDataset(
            dataset_df = self.val_dataset,
            filename_col_name = self.filename_col_name, 
            label_col_name = self.label_col_name,
            transform = self.test_val_transform
        )

        test_dataset = DoggoDataset(
            dataset_df = self.test_dataset,
            filename_col_name = self.filename_col_name, 
            label_col_name = self.label_col_name,
            transform = self.test_val_transform
        )

        # Create data loaders.
        num_workers = int(os.cpu_count())
        print(f"num_worker: {num_workers}")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers = num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers = num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers = num_workers)

        return train_dataloader, val_dataloader, test_dataloader
    
class DataUtils:
    @staticmethod
    def get_label_dict_n_relabel_to_int(dataset_df: pd.DataFrame,
                                    label_col_name: str,
                                    filenames_col_name: str):
        
        # Check that the specified columns exist in the DataFrame
        if label_col_name not in dataset_df.columns:
            raise ValueError(f"Column '{label_col_name}' does not exist in the DataFrame.")
        if filenames_col_name not in dataset_df.columns:
            raise ValueError(f"Column '{filenames_col_name}' does not exist in the DataFrame.")
        
        # get_label dict
        unique_labels = np.sort(dataset_df["label"].unique()).tolist()
        label_dict = {val:i for i, val in enumerate(unique_labels)}

        # relabel to int
        dataset_df[label_col_name] = dataset_df[label_col_name].apply(lambda x: label_dict[x])

        # Flip label dict back
        label_dict = {i:val.replace("_", " ").title() for val, i in label_dict.items()}

        return dataset_df,label_dict

    @staticmethod
    def get_random_filenames_by_label(dataset_df: pd.DataFrame,
                                    label_col_name: str,
                                    filenames_col_name: str,
                                    n_labels: int,
                                    n_samples: int) -> pd.DataFrame:
        """
        Returns a DataFrame containing up to n_samples filenames for each of n_labels unique labels.

        Args:
            dataset_df (pandas.DataFrame): The DataFrame containing the columns.
            label_col_name (str): The name of the column containing the labels.
            filenames_col_name (str): The name of the column containing the filenames.
            n_labels (int): The number of unique labels to select.
            n_samples (int): The maximum number of filenames to select for each label.

        Returns:
            pandas.DataFrame: A DataFrame containing up to n_samples filenames for each of n_labels unique labels.
            dictionary: Mapping numerical labels to actual names
            
        Raises:
            ValueError: If the specified label column name does not exist in the DataFrame.
            ValueError: If the specified filenames column name does not exist in the DataFrame.
        """
        # Check that the specified columns exist in the DataFrame
        if label_col_name not in dataset_df.columns:
            raise ValueError(f"Column '{label_col_name}' does not exist in the DataFrame.")
        if filenames_col_name not in dataset_df.columns:
            raise ValueError(f"Column '{filenames_col_name}' does not exist in the DataFrame.")

        # Get a list of unique labels in the label column
        unique_labels = dataset_df[label_col_name].unique().tolist()

        # Check that there are enough unique labels in the column
        if len(unique_labels) < n_labels:
            print(f"Dataframe '{label_col_name}' contains fewer than {n_labels} unique labels, using {len(unique_labels)}")
            n_labels = len(unique_labels)

        # Randomly select n_labels unique labels
        selected_labels = random.sample(unique_labels, n_labels)

        selected_filenames = pd.DataFrame(columns = dataset_df.columns)

        for selected_label in selected_labels:

            tmp = dataset_df[dataset_df[label_col_name] == selected_label]

            if len(tmp) >= n_samples:
                tmp = tmp.sample(n=n_samples, random_state = env_vars.random_state)
            
            selected_filenames = pd.concat([selected_filenames, tmp], ignore_index=True)
        
        selected_filenames.reset_index(drop = True, inplace= True)

        return DataUtils.get_label_dict_n_relabel_to_int(selected_filenames,label_col_name, filenames_col_name)
    
    @staticmethod
    def train_val_test_split(dataset_df: pd.DataFrame,
                             split_ratio: float,
                             label_col_name: str):
        
        # shuffle data frame
        train_dataset_df, test_dataset_df = train_test_split(
            dataset_df,
            test_size = split_ratio,
            stratify=dataset_df[label_col_name],
            random_state=env_vars.random_state)
        
        train_dataset_df, val_dataset_df = train_test_split(
            train_dataset_df,
            test_size = split_ratio,
            stratify=train_dataset_df[label_col_name],
            random_state=env_vars.random_state)
        
        return train_dataset_df, val_dataset_df, test_dataset_df