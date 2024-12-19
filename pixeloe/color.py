import cv2
import numpy as np
from PIL import Image


def match_color(source, target, level=5):
    source_color = source[:, :, :3]  # Extract BGR
    alpha = source[:, :, 3]  # Extract alpha channel (unchanged)
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
    # Use wavelet colorfix method to match original low-frequency data
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

    # Combine matched color channels with the original alpha channel
    output = cv2.merge((matched_color, alpha))

    return output


def wavelet_colorfix(inp, target, level=5):
    inp_high, _ = wavelet_decomposition(inp, level)
    _, target_low = wavelet_decomposition(target, level)
    output = inp_high + target_low
    return output


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


def color_quant(image, colors=32, weights=None, repeats=64, method="kmeans"):
    # Handle RGBA images
    has_alpha = image.shape[2] == 4

    # Separate RGB and alpha channels
    if has_alpha:
        rgb_image = image[:, :, :3]
        alpha_channel = image[:, :, 3]
    else:
        rgb_image = image

    match method:
        case "kmeans":
            if weights is not None:
                h, w, c = rgb_image.shape
                pixels = []
                weights = weights / np.max(weights) * repeats
                for i in range(h):
                    for j in range(w):
                        repeat_times = max(1, int(weights[i, j]))
                        pixels.extend([rgb_image[i, j]] * repeat_times)
                pixels = np.array(pixels, dtype=np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 32, 1)
                _, labels, palette = cv2.kmeans(
                    pixels, colors, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS
                )

                quantized_rgb = np.zeros((h, w, c), dtype=np.uint8)
                label_idx = 0
                for i in range(h):
                    for j in range(w):
                        repeat_times = max(1, int(weights[i, j]))
                        quantized_rgb[i, j] = palette[labels[label_idx]]
                        label_idx += repeat_times
            else:
                h, w, c = rgb_image.shape
                pixels = rgb_image.reshape((-1, 3)).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 32, 1)
                _, labels, palette = cv2.kmeans(
                    pixels, colors, None, criteria, 4, cv2.KMEANS_RANDOM_CENTERS
                )

                quantized_rgb = np.zeros((h, w, c), dtype=np.uint8)
                for i in range(h):
                    for j in range(w):
                        quantized_rgb[i, j] = palette[labels[i * w + j]]

            # If the image has an alpha channel, combine it with the quantized RGB
            if has_alpha:
                quantized_image = cv2.merge((quantized_rgb, alpha_channel))
            else:
                quantized_image = quantized_rgb

            return quantized_image

        case "maxcover":
            img = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_quant = img_pil.quantize(colors, 1, kmeans=colors).convert("RGB")
            quantized_rgb = cv2.cvtColor(np.array(img_quant), cv2.COLOR_RGB2BGR)

            # If the image has an alpha channel, combine it with the quantized RGB
            if has_alpha:
                quantized_image = cv2.merge((quantized_rgb, alpha_channel))
            else:
                quantized_image = quantized_rgb

            return quantized_image
