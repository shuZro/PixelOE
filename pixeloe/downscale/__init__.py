from .conventional import bicubic, nearest
from .k_centroid import k_centroid_downscale


downscale_mode = {
    "bicubic": bicubic,
    "nearest": nearest,
    "k-centroid": k_centroid_downscale,
}
