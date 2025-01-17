from time import time

import cv2
import numpy
import numpy as np
from PIL import Image

from .color import match_color, color_styling, color_quant
from .downscale import downscale_mode
from .outline import outline_expansion, expansion_weight
from .utils import isiterable


def pixelize(
        img: Image,
        mode="nearest",
        target_size=128,
        patch_size=16,
        thickness=2,
        color_matching=True,
        contrast=1.0,
        saturation=1.0,
        colors=16,
):
    img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGBA2BGRA)
    H, W, _ = img.shape

    ratio = W / H
    if isiterable(target_size) and len(target_size) > 1:
        target_org_hw = tuple([int(i * patch_size) for i in target_size][:2])
        ratio = target_org_hw[0] / target_org_hw[1]
        target_org_size = target_org_hw[1]
        target_size = ((target_org_size ** 2) / (patch_size ** 2) * ratio) ** 0.5
    else:
        if isiterable(target_size):
            target_size = target_size[0]
        target_org_size = (target_size ** 2 * patch_size ** 2 / ratio) ** 0.5
        target_org_hw = (int(target_org_size * ratio), int(target_org_size))

    img = cv2.resize(img, target_org_hw)
    org_img = img.copy()

    if thickness:
        img, weight = outline_expansion(img, thickness, thickness, patch_size, 9, 4)

    if color_matching:
        img = match_color(img, org_img)

    img_sm = downscale_mode[mode](img, target_size)

    if colors is not None:
        img_sm_orig = img_sm.copy()

        img_sm = color_quant(
            img_sm,
            colors,
        )
        img_sm = match_color(img_sm, img_sm_orig, 3)

    if contrast != 1 or saturation != 1:
        img_sm = color_styling(img_sm, saturation, contrast)

    image = cv2.cvtColor(img_sm, cv2.COLOR_BGRA2RGBA)
    image = Image.fromarray(image)
    image = remove_non_fully_transparent_pixels(image)

    return image


def remove_non_fully_transparent_pixels(image):
    image = image.convert('RGBA')
    pixels = image.load()

    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = pixels[x, y]
            if a > 0 and a < 150:
                pixels[x, y] = (r, g, b, 0)
            if a > 150 and a < 255:
                pixels[x, y] = (r, g, b, 255)

    return image


if __name__ == "__main__":
    t0 = time()
    img = cv2.imread("img/house.png")
    t1 = time()
    img = pixelize(img, target_size=128, patch_size=8)
    t2 = time()
    cv2.imwrite("test.png", img)
    t3 = time()

    print(f"read time: {t1 - t0:.3f}sec")
    print(f"pixelize time: {t2 - t1:.3f}sec")
    print(f"write time: {t3 - t2:.3f}sec")
