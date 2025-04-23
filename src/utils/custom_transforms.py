import umap
import umap.plot
import wandb
import math
from .helpfuns import *
from .dist_utills import *
from PIL import ImageOps, ImageFilter, ImageFile
from torch import nn, randn, Tensor, FloatTensor
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms import Compose,RandomApply, InterpolationMode
import torchvision.transforms.functional as F


# ================================= Need to apply rand_apply ================================= #

class ResizeMultiChannels(nn.Module):
    """Resizes a multichannel image."""
    def __init__(self, size):
        """
        Args:
            size(int or tuple): size of 
        """
        self.size = size
    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be resized.

        Returns:
            Resized image (torch.Tensor).
        """
        tensor_img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        resized_tensor = F.resize(tensor_img, self.size)
        resized_img = resized_tensor.numpy().transpose(1, 2, 0)
        return resized_img

class CenterCropMultiChannels(nn.Module):
    """Performs center cropping on a multichannel image."""
    def __init__(self, size):
        """
        Args:
            size (int or tuple): Desired output size of the crop. If size is an int,
                a square crop of size (size, size) is returned. If size is a tuple,
                it should contain (height, width).
        """
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be center cropped.

        Returns:
            Center cropped image (numpy.ndarray).
        """
        tensor_img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        cropped_tensor = F.center_crop(tensor_img, self.size)
        cropped_img = cropped_tensor.numpy().transpose(1, 2, 0)
        return cropped_img

class VerticalFlipMultiChannels(nn.Module):
    """Performs a vertical flip on a multichannel image."""
    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be vertically flipped.

        Returns:
            Vertically flipped image (numpy.ndarray).
        """
        tensor_img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        flipped_tensor = F.vflip(tensor_img)
        flipped_img = flipped_tensor.numpy().transpose(1, 2, 0)
        return flipped_img

class HorizontalFlipMultiChannels(nn.Module):
    """Performs a horizontal flip on a multichannel image."""
    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be horizontally flipped.

        Returns:
            Vertically flipped image (numpy.ndarray).
        """
        tensor_img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        flipped_tensor = F.hflip(tensor_img)
        flipped_img = flipped_tensor.numpy().transpose(1, 2, 0)
        return flipped_img

class RotationMultiChannels(nn.Module):
    """Performs a random rotation on a multichannel image."""
    def __init__(self, degrees, resample=False, expand=False):
        """
        Args:
            degrees (sequence or number): Range of degrees to select from.
                If degrees is a number, the range of degrees to select from
                will be (-degrees, +degrees).
            resample (bool, optional): If False, the default, the input image
                will be rotated using pixel area resampling. If True, the
                input image will be rotated using bilinear interpolation.
            expand (bool, optional): If True, the output image size will be
                expanded to include the whole rotated image. If False, the
                output image size will be the same as the input image.
                Expanding the size can help avoid cropping out parts of the
                image during rotation.
        """
        self.degrees = degrees
        self.resample = resample
        self.expand = expand

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be randomly rotated.

        Returns:
            Randomly rotated image (numpy.ndarray).
        """
        tensor_img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        if self.resample:
            rotated_tensor = F.rotate(img=tensor_img, angle=self.degrees, interpolation=InterpolationMode.BILINEAR, expand=self.expand)
        else:
            rotated_tensor = F.rotate(img=tensor_img, angle=self.degrees, expand=self.expand)

        rotated_img = rotated_tensor.numpy().transpose(1, 2, 0)
        return rotated_img

class CropMultiChannels(nn.Module):
    """Crop the given numpy array at a random location."""

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for random crop.

        Args:
            img (torch.Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to crop.
        """
        h, w = img.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be cropped.

        Returns:
            numpy.ndarray: Cropped image.
        """
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()

        if self.padding is not None:
            img_tensor = F.pad(img_tensor, self.padding, self.fill, self.padding_mode)

        # Pad the width if needed
        if self.pad_if_needed and img_tensor.shape[-1] < self.size[1]:
            img_tensor = F.pad(img_tensor, (0, self.size[1] - img_tensor.shape[-1]), self.fill, self.padding_mode)
        # Pad the height if needed
        if self.pad_if_needed and img_tensor.shape[-2] < self.size[0]:
            img_tensor = F.pad(img_tensor, (0, self.size[0] - img_tensor.shape[-2]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img_tensor, self.size)

        cropped_tensor = F.crop(img_tensor, i, j, h, w)
        cropped_img = cropped_tensor.numpy().transpose(1, 2, 0)
        return cropped_img
    
class GaussianBlurMultiChannels(nn.Module):
    """Apply random Gaussian blur to each channel of a multichannel image."""

    def __init__(self, radius_min=0.1, radius_max=2.0):
        """
        Args:
            radius_min (float): Minimum blur radius. Default is 0.1.
            radius_max (float): Maximum blur radius. Default is 2.0.
        """
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Input image with shape (height, width, channels).

        Returns:
            numpy.ndarray: Image with Gaussian blur applied to each channel.
        """
        img_tensor = torch.from_numpy(img)  # Convert numpy array to torch tensor
        img_tensor = img_tensor.permute(2, 0, 1)  # Change channel dimension to the first dimension

        n_channels = img_tensor.shape[0]

        for ind in range(n_channels):
            radius = torch.FloatTensor(1).uniform_(self.radius_min, self.radius_max)
            img_tensor[ind] = self.apply_gaussian_blur(img_tensor[ind], radius)

        img_tensor = img_tensor.permute(1, 2, 0)  # Change channel dimension back to the last dimension
        return img_tensor.numpy()  # Convert back to numpy array

    @staticmethod
    def apply_gaussian_blur(channel, radius):
        """
        Apply Gaussian blur to a single channel.

        Args:
            channel (torch.Tensor): Single channel image tensor.
            radius (float): Blur radius.

        Returns:
            torch.Tensor: Blurred channel tensor.
        """
        # Convert channel to 2D tensor (height, width)
        channel_2d = channel.unsqueeze(0)

        # Convert radius to kernel size
        kernel_size = int(2 * radius + 1)

        # Create Gaussian kernel
        kernel = GaussianBlurMultiChannels.create_gaussian_kernel(kernel_size, radius)

        # Apply Gaussian blur using torch.nn.conv2d with padding='same'
        blurred_channel_2d = nn.Conv2d(
            channel_2d.unsqueeze(0),
            kernel.unsqueeze(0),
            kernel_size=kernel_size,
            padding=radius,
            stride=1,
            groups=1,
        )

        return blurred_channel_2d.squeeze(0).squeeze(0)

    @staticmethod
    def create_gaussian_kernel(kernel_size, sigma):
        """
        Create a Gaussian kernel of the specified size and sigma.

        Args:
            kernel_size (int): Size of the kernel (both height and width).
            sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
            torch.Tensor: Gaussian kernel tensor.
        """
        # Create a 1D Gaussian kernel
        kernel_1d = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
        kernel_1d = kernel_1d / torch.sum(kernel_1d)

        # Expand the 1D kernel to 2D
        kernel = torch.outer(kernel_1d, kernel_1d)

        return kernel


class GaussianNoiseMultiChannels(nn.Module):
    """
    Adds Gaussian noise to a numpy array image.

    Arguments
    ---------
    mean (float): Mean of the Gaussian distribution. Default is 0.0.
    std (float): Standard deviation of the Gaussian distribution. Default is 0.05.

    Returns
    -------
    numpy.ndarray: Image with Gaussian noise added.
    """

    def __init__(self, mean=0.0, std=0.05):
        self.std = std
        self.mean = mean
     
    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Input image with shape (height, width, channels).

        Returns:
            numpy.ndarray: Image with Gaussian noise added.
        """
        noise = np.random.normal(self.mean, self.std, size=img.shape)
        return img + noise
  
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
 
class ChannelIntensityShift(nn.Module):
    """
    Shifts the intensity of each channel of a numpy array image by a random value in the range [-0.3, 0.3].

    Arguments
    ---------
    shift_range (tuple): Range of values to shift the intensity of each channel by. Default is (-0.3, 0.3).

    Returns
    -------
    numpy.ndarray: Image with intensity shifted.
    """

    def __init__(self, shift_range=(-0.3, 0.3)):
        self.shift_range = shift_range

    def __call__(self, img):
        img_tensor = torch.from_numpy(img)  # Convert numpy array to torch tensor
        for i in range(img_tensor.shape[0]):  # Loop over the channels
            shift_value = torch.FloatTensor(1).uniform_(*self.shift_range)
            img_tensor[i] += shift_value

        return img_tensor.numpy()  # Convert back to numpy array

class ChannelBrightnessShift(nn.Module):
    """
    Brightens each channel of a numpy array image by a random value in the range [0.5, 1.5].

    Arguments
    ---------
    brightness_range (tuple): Range of values to brighten each channel by. Default is (0.5, 1.5).

    Returns
    -------
    numpy.ndarray: Image with brightness adjusted.
    """
    
    def __init__(self, brightness_range=(0.5, 1.5)):
        self.brightness_range = brightness_range

    def __call__(self, img):
        img_tensor = torch.from_numpy(img)  # Convert numpy array to torch tensor
        for i in range(img_tensor.shape[2]):  # Loop over the channels (assuming img_tensor shape is (height, width, channels))
            brightness_factor = torch.FloatTensor(1).uniform_(*self.brightness_range)
            img_tensor[:, :, i] *= brightness_factor

        return img_tensor.numpy()  # Convert back to numpy array

class ChannelContrastShift(nn.Module):
    """
    Apply random contrast adjustment to each channel of a multichannel image.
    """

    def __init__(self, p=0.2):
        """
        Args:
            p (float): Probability of applying contrast adjustment. Default is 0.2.
        """
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Input image with shape (height, width, channels).

        Returns:
            numpy.ndarray: Image with contrast adjustment applied to each channel.
        """
        if img.max() == 0:
            return img.astype(np.float32)
        
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
        n_channels = img_tensor.shape[0]
        
        for ind in range(n_channels):
            factor = max(np.random.normal(1, self.p), 0.5)
            img_tensor[ind] = self.adjust_contrast(img_tensor[ind], factor)
        
        return img_tensor.numpy().transpose((1, 2, 0)).astype(np.float16)

    @staticmethod
    def adjust_contrast(img, factor):
        """
        Adjusts the contrast of an image tensor using adjust_contrast.

        Args:
            img (torch.Tensor): Input image tensor.
            factor (float): Contrast adjustment factor.

        Returns:
            torch.Tensor: Image tensor with contrast adjustment applied.
        """
        mean = torch.mean(img)
        img = (img - mean) * factor + mean
        return torch.clamp(img, 0.0, 1.0)
    
# ================================= Others ================================= #
class ResizedCropMultiChannels(nn.Module):
    """Crop a random portion of multichannel numpy array and resize it to a given size."""

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation="bilinear", padding=None):
        """
        Args:
            size (int or tuple): Size of the output image. If size is an int, a square image (size, size) is returned.
                If size is a tuple, it should contain (height, width).
            scale (tuple): Range of the random size of the cropped image compared to the original image.
            ratio (tuple): Range of the random aspect ratio of the cropped image.
            interpolation (str, optional): Interpolation method for resizing. Default is "bilinear".
            padding (int or tuple, optional): Optional padding on each border of the image.
                If padding is a single number, pad all borders with the same value.
                If padding is a tuple of length 4, pad the left, top, right, and bottom borders with the respective values.
        """
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.padding = padding
        if interpolation == 'bilinear':
            self.interpolation = InterpolationMode.BILINEAR
        elif interpolation == 'bicubic':
            self.interpolation = InterpolationMode.BICUBIC

    def get_params(self, img):
        """Get parameters for random resizing and cropping.

        Args:
            img (numpy.ndarray): Image to be cropped.

        Returns:
            tuple: params (i, j, h, w, size) to be passed to crop.
        """
        h, w = img.shape[:2]

        area = h * w

        for attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            w_ratio = int(round(np.sqrt(target_area * aspect_ratio)))
            h_ratio = int(round(np.sqrt(target_area / aspect_ratio)))

            if self.padding is not None:
                padded_w = w + 2 * self.padding if self.padding > 0 else w
                padded_h = h + 2 * self.padding if self.padding > 0 else h
                if padded_h >= h_ratio and padded_w >= w_ratio:
                    i = random.randint(0, padded_h - h_ratio) if padded_h - h_ratio > 0 else 0
                    j = random.randint(0, padded_w - w_ratio) if padded_w - w_ratio > 0 else 0
                    return i, j, h_ratio, w_ratio, self.size

            else:
                if h >= h_ratio and w >= w_ratio:
                    i = random.randint(0, h - h_ratio) if h - h_ratio > 0 else 0
                    j = random.randint(0, w - w_ratio) if w - w_ratio > 0 else 0
                    return i, j, h_ratio, w_ratio, self.size

        # Fallback to central crop
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.ratio):
            w_ratio = w
            h_ratio = int(round(w_ratio / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h_ratio = h
            w_ratio = int(round(h_ratio * max(self.ratio)))
        else:  # Whole image
            w_ratio = w
            h_ratio = h

        i = (h - h_ratio) // 2
        j = (w - w_ratio) // 2
        return i, j, h_ratio, w_ratio, self.size

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray): Image to be cropped and resized.

        Returns:
            numpy.ndarray: Cropped and resized image.
        """
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()

        if self.padding is not None:
            img_tensor = F.pad(img_tensor, self.padding, 0)

        i, j, h, w, size = self.get_params(img_tensor)

        # Ensure the crop is squared by using the minimum size
        crop_size = min(h, w)
        img_tensor = F.resized_crop(img_tensor, i, j, crop_size, crop_size, size, self.interpolation)

        cropped_resized_img = img_tensor.numpy().transpose(1, 2, 0)
        return cropped_resized_img

class MultiCrop(nn.Module):
    """
    Creates multiple crops of an image (mostly useful for self-supervision pipeline)

    Arguments
    ---------
    n_crops (list): number of crops to create
    sizes (list): size of each crop
    scales (list): scale of each crop

    Returns
    -------
    images (list): list of images
    """
    
    def __init__(self, n_crops:list, sizes:list, scales):
        self.n_crops = n_crops
        self.sizes = sizes
        self.scales = scales
        self.crop_augs = []
        for n, s in enumerate(n_crops):
            size = sizes[n]
            scale = scales[n]
            for _ in range(s):
                self.crop_augs.append(
                    Compose([
                        ResizedCropMultiChannels(
                            size=size, scale=scale, interpolation="bicubic"
                        ),
                        rand_apply(ChannelBrightnessShift(),p=0.8),
                        rand_apply(ChannelIntensityShift(), p=0.8),
                        rand_apply(GaussianNoiseMultiChannels(), p=0.2),
                        rand_apply(HorizontalFlipMultiChannels(), p=0.5),
                        ])
                )

    def __call__(self, image, augmentations):
        images = []
        for c_aug in self.crop_augs:
            images.append(augmentations(c_aug(image)))

        if not images:
            return image
        return images

def rand_apply(tranform, p=0.5):
    """
    Apply a transform with probability p

    Arguments
    ---------
    tranform (torchvision.transforms): transform to apply

    Returns
    -------
    transform (torchvision.transforms): transform with probability p
    """
    return RandomApply(torch.nn.ModuleList([tranform]), p)