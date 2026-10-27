import cv2
import numpy as np
from numba import njit

def rescale_image(image,rescale_mode,max_pixels,max_fraction, is_mask=False):
    h, w = image.shape[0], image.shape[1]
    match rescale_mode:
        case "Disabled":
            return image
        case "Pixels":
            max_pixels = int(max_pixels)
            max_size = max(h, w)
            fraction = max_pixels / max_size
        case "Fraction":
            fraction = float(max_fraction)
    new_h = max(1, int(h * fraction))
    new_w = max(1, int(w * fraction))

    if new_h * new_w> h * w:
        # Upscaling
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    else:
        # Downscaling
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA

    image = cv2.resize(image, (new_w,new_h), interpolation=interpolation)
    return image

def normalize_image(image: np.ndarray, margin,lower_quantile,upper_quantile) -> np.ndarray:
    """
    image: np.ndarray of type float with format Z, Y, X or Y, X
    returns: np.ndarray of type float normalized between 0 and 1
    """
    shape = np.array(image.shape)
    offset = (shape * margin).astype(int)
    cropped = image[..., offset[-2]:-offset[-2], offset[-1]:-offset[-1]]

    hist = _numba_histogram(cropped)
    cdf = np.cumsum(hist) / cropped.size

    min_val = float(np.searchsorted(cdf, lower_quantile))
    max_val = float(np.searchsorted(cdf, upper_quantile))

    diff = max_val - min_val
    if diff > 0:
        if image.ndim == 3:
            _numba_normalize_inplace_3d(image, float(min_val), float(diff))
        else:
            _numba_normalize_inplace_2d(image, float(min_val), float(diff))
    else:
        image.fill(0.0)

    return image


@njit(cache=True, nogil=True)
def _numba_histogram(arr):
    hist = np.zeros(65536, dtype=np.uint64)
    for val in arr.flat:
        idx = int(val)
        if 0 <= idx <= 65535:
            hist[idx] += 1

    return hist

@njit(cache=True, nogil=True)
def _numba_normalize_inplace_3d(arr, min_val, diff):
    d0, d1, d2 = arr.shape
    for z in range(d0):
        for y in range(d1):
            for x in range(d2):
                val = (arr[z, y, x] - min_val) / diff
                if val < 0.0:
                    arr[z, y, x] = 0.0
                elif val > 1.0:
                    arr[z, y, x] = 1.0
                else:
                    arr[z, y, x] = val

@njit(cache=True, nogil=True)
def _numba_normalize_inplace_2d(arr, min_val, diff):
    d0, d1 = arr.shape
    for y in range(d0):
        for x in range(d1):
            val = (arr[y, x] - min_val) / diff
            if val < 0.0:
                arr[y, x] = 0.0
            elif val > 1.0:
                arr[y, x] = 1.0
            else:
                arr[y, x] = val

# ==========================================
# NUMBA WARM-UP
# ==========================================

_dummy_float32_3d = np.zeros((3, 10, 10), dtype=np.float32)
_dummy_float32_2d = np.zeros((10, 10), dtype=np.float32)

_numba_histogram(_dummy_float32_3d[..., 1:-1, 1:-1])
_numba_histogram(_dummy_float32_2d[..., 1:-1, 1:-1])
_numba_histogram(_dummy_float32_3d)
_numba_histogram(_dummy_float32_2d)

_numba_normalize_inplace_3d(_dummy_float32_3d, 0.0, 1.0)
_numba_normalize_inplace_2d(_dummy_float32_2d, 0.0, 1.0)

del _dummy_float32_3d, _dummy_float32_2d