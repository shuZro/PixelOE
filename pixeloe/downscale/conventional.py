import cv2
import numpy as np
from PIL import Image

from PIL import Image
import numpy as np
import cv2


def nearest(img, target_size=128):
    """
    Resizes an image to the target size using the nearest-neighbor method, preserving the alpha channel.

    Parameters:
        img (numpy.ndarray): Input image with RGBA channels (H, W, 4).
        target_size (int): The target size for the resized image (size for the shortest dimension).

    Returns:
        numpy.ndarray: Resized image with RGBA channels.
    """
    H, W, C = img.shape

    # Ensure that the image has an alpha channel
    if C != 4:
        raise ValueError("Input image must have 4 channels (RGBA).")

    # Calculate aspect ratio and target dimensions
    ratio = W / H
    target_size = (target_size ** 2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))

    # Split the channels (RGBA)
    img_rgb = img[:, :, :3]  # RGB channels
    img_alpha = img[:, :, 3]  # Alpha channel

    # Convert RGB to PIL Image for resizing
    img_rgb_pil = Image.fromarray(img_rgb)
    img_alpha_pil = Image.fromarray(img_alpha)

    # Resize using nearest-neighbor
    img_rgb_resized = img_rgb_pil.resize(target_hw, Image.NEAREST)
    img_alpha_resized = img_alpha_pil.resize(target_hw, Image.NEAREST)

    # Convert back to numpy arrays
    img_rgb_resized = np.asarray(img_rgb_resized)
    img_alpha_resized = np.asarray(img_alpha_resized)

    # Combine resized RGB and alpha channels
    img_resized = np.dstack((img_rgb_resized, img_alpha_resized))

    return img_resized


def bicubic(img, target_size=128):
    """
    Resizes an image to the target size using bicubic interpolation, preserving the alpha channel.

    Parameters:
        img (numpy.ndarray): Input image with RGBA channels (H, W, 4).
        target_size (int): The target size for the resized image (size for the shortest dimension).

    Returns:
        numpy.ndarray: Resized image with RGBA channels.
    """
    H, W, C = img.shape

    # Ensure the input image has an alpha channel
    if C != 4:
        raise ValueError("Input image must have 4 channels (RGBA).")

    # Calculate aspect ratio and target dimensions
    ratio = W / H
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))

    # Split the image into RGB and alpha channels
    img_rgb = img[:, :, :3]  # RGB channels
    img_alpha = img[:, :, 3]  # Alpha channel

    # Convert RGB and alpha to PIL Images
    img_rgb_pil = Image.fromarray(img_rgb)
    img_alpha_pil = Image.fromarray(img_alpha)

    # Resize both RGB and alpha channels using bicubic interpolation
    img_rgb_resized = img_rgb_pil.resize(target_hw, Image.BICUBIC)
    img_alpha_resized = img_alpha_pil.resize(target_hw, Image.BICUBIC)

    # Convert resized PIL Images back to numpy arrays
    img_rgb_resized = np.asarray(img_rgb_resized)
    img_alpha_resized = np.asarray(img_alpha_resized)

    # Combine resized RGB and alpha channels
    img_resized = np.dstack((img_rgb_resized, img_alpha_resized))

    return img_resized
