import cv2
import numpy as np
from PIL import Image


def match_color(source, target, level=5, quant_colors=32):
    source_color = source[:, :, :3]  # Extract BGR
    alpha = source[:, :, 3]  # Extract alpha channel
    target_color = target[:, :, :3]

    # Convert BGR to L*a*b* and match std/mean
    source_lab = cv2.cvtColor(source_color, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    target_lab = cv2.cvtColor(target_color, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    result = (source_lab - np.mean(source_lab)) / np.std(source_lab)
    result = result * np.std(target_lab) + np.mean(target_lab)
    matched_color = cv2.cvtColor(
        (result * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )

    matched_color = matched_color.astype(np.float32)
    matched_color[:, :, 0] = wavelet_colorfix(
        matched_color[:, :, 0], target_color[:, :, 0], level=level
    )
    matched_color[:, :, 1] = wavelet_colorfix(
        matched_color[:, :, 1], target_color[:, :, 1], level=level
    )
    matched_color[:, :, 2] = wavelet_colorfix(
        matched_color[:, :, 2], target_color[:, :, 2], level=level
    )
    matched_color = matched_color.clip(0, 255).astype(np.uint8)

    # Combine matched color with alpha channel
    output = cv2.merge((matched_color, alpha))

    # Enforce solid colors using quantization
    output = color_quant(output, colors=quant_colors)

    return output




def wavelet_colorfix(inp, target, level=3):  # Reduced levels for speed
    inp_high, _ = wavelet_decomposition(inp, level)
    _, target_low = wavelet_decomposition(target, level)
    return inp_high + target_low



def wavelet_decomposition(inp, levels):
    high_freq = np.zeros_like(inp)
    for i in range(1, levels + 1):
        radius = 2 ** i
        low_freq = wavelet_blur(inp, radius)
        high_freq = high_freq + (inp - low_freq)
        inp = low_freq
    return high_freq, low_freq


def wavelet_blur(inp, radius):
    kernel_size = 2 * radius + 1
    output = cv2.GaussianBlur(inp, (kernel_size, kernel_size), 0)
    return output


def color_styling(inp, saturation=1.2, contrast=1.1):
    output = inp.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    output[:, :, 1] = output[:, :, 1] * saturation
    output[:, :, 2] = output[:, :, 2] * contrast - (contrast - 1)
    output = np.clip(output, 0, 1)
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    return output


def color_quant(image, colors=32):
    has_alpha = image.shape[2] == 4

    # Separate RGB and alpha channels
    if has_alpha:
        rgb_image = image[:, :, :3]
        alpha_channel = image[:, :, 3]
    else:
        rgb_image = image

    # Flatten RGB channels for k-means clustering
    h, w, c = rgb_image.shape
    pixels = rgb_image.reshape((-1, 3)).astype(np.float32)

    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, palette = cv2.kmeans(
        pixels, colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    quantized_rgb = palette[labels.flatten()].reshape(h, w, 3).astype(np.uint8)

    # Combine quantized RGB with alpha if present
    if has_alpha:
        quantized_image = cv2.merge((quantized_rgb, alpha_channel))
    else:
        quantized_image = quantized_rgb

    return quantized_image

