from numba import njit
import numpy as np

def rgb_to_hex(rgb_color):
    """
    Converts a rgb color to hex color
    Args:
        rgb_color (tuple)
    Returns:
        hex_color (str)
    """
    return "#{:02x}{:02x}{:02x}".format(*rgb_color)

@njit(cache=True, nogil=True)
def _numba_shift_mask(mask_flat, outline_flat):
    seen = np.zeros(65536, dtype=np.bool_)
    max_id = 0

    for i in range(outline_flat.size):
        val = outline_flat[i]
        if val != 0:
            seen[val] = True
            if val > max_id:
                max_id = val

    if max_id == 0:
        return False, 0, np.zeros(1, dtype=np.uint16)

    unique_count = 0
    for i in range(1, max_id + 1):
        if seen[i]:
            unique_count += 1

    if max_id == unique_count:
        return False, 0, np.zeros(1, dtype=np.uint16)

    lookup = np.zeros(max_id + 1, dtype=np.uint16)
    new_id = 1
    for i in range(1, max_id + 1):
        if seen[i]:
            lookup[i] = new_id
            new_id += 1

    for i in range(mask_flat.size):
        val_m = mask_flat[i]
        if val_m != 0 and val_m <= max_id:
            mask_flat[i] = lookup[val_m]

        val_o = outline_flat[i]
        if val_o != 0 and val_o <= max_id:
            outline_flat[i] = lookup[val_o]

    return True, max_id, lookup


def mask_shifting(mask_data):
    """
    Shifts the mask when a mask got deleted or added to restore an order without gaps.
    """
    mask = mask_data["masks"]
    outline = mask_data["outlines"]

    shifted, max_id, lookup = _numba_shift_mask(mask.ravel(), outline.ravel())

    if not shifted:
        return None

    mapping_dict = {}
    for old_id in range(1, max_id + 1):
        new_id = lookup[old_id]
        if new_id != 0:
            mapping_dict[old_id] = int(new_id)

    return mapping_dict

def search_free_id(mask,outline, slice_id):
    """
    Search in a NumPy array of integers (e.g., [1,1,2,2,3,4,5,5,7,7]) for the first missing number (in this case, 6).
    If no gap is found, return the highest value + 1.
    """

    max_val = max(mask.max(), outline.max())
    if max_val == 0:
        return 1

    if mask.ndim == 3 and slice_id <0:
        return int(max_val + 1)

    counts = np.bincount(mask.ravel(), minlength=max_val + 1)
    counts += np.bincount(outline.ravel(), minlength=max_val + 1)

    zero_indices = np.where(counts[1:] == 0)[0]

    if zero_indices.size > 0:
        return int(zero_indices[0] + 1)
    else:
        return int(max_val + 1)


@njit(cache=True, nogil=True)
def _numba_count(arr):
    seen = np.zeros(65536, dtype=np.bool_)
    count = 0
    for val in arr.flat:
        if val != 0 and not seen[val]:
            seen[val] = True
            count += 1
    return count

def count_ids(mask_array, current_slice):
    if mask_array.ndim == 3 and current_slice >= 0:
        return _numba_count(mask_array[current_slice])
    else:
        return _numba_count(mask_array)


@njit(cache=True, nogil=True)
def _numba_process_2d_slice(mask_slice):
    h, w = mask_slice.shape
    outline = np.zeros((h, w), dtype=np.uint16)

    for y in range(h):
        for x in range(w):
            if mask_slice[y, x] == 0:
                found_id = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            cid = mask_slice[ny, nx]
                            if cid > 0:
                                if cid > found_id:
                                    found_id = cid

                if found_id > 0:
                    outline[y, x] = found_id

    return outline


@njit(cache=True, nogil=True)
def _numba_build_canvas(mask_slice, outline_slice, image_mask, m_b, m_g, m_r, opacity, o_b, o_g, o_r):
    h, w = mask_slice.shape
    for y in range(h):
        for x in range(w):
            if outline_slice[y, x] > 0:
                image_mask[y, x, 0] = o_b
                image_mask[y, x, 1] = o_g
                image_mask[y, x, 2] = o_r
                image_mask[y, x, 3] = 255
            elif mask_slice[y, x] > 0:
                image_mask[y, x, 0] = m_b
                image_mask[y, x, 1] = m_g
                image_mask[y, x, 2] = m_r
                image_mask[y, x, 3] = opacity


@njit(cache=True, nogil=True)
def _numba_bbox_for_ids_3d(mask, outline, ids_to_delete):
    lookup = np.zeros(65536, dtype=np.bool_)
    for c in ids_to_delete:
        lookup[c] = True

    z_min, y_min, x_min = 999999, 999999, 999999
    z_max, y_max, x_max = -1, -1, -1

    d0, d1, d2 = mask.shape
    for z in range(d0):
        for y in range(d1):
            for x in range(d2):
                m = mask[z, y, x]

                if m != 0 and lookup[m]:
                    if z < z_min: z_min = z
                    if z > z_max: z_max = z
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x
                    continue

                o = outline[z, y, x]
                if o != 0 and lookup[o]:
                    if z < z_min: z_min = z
                    if z > z_max: z_max = z
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x

    if z_max == -1: return (0, 0, 0, 0, 0, 0)
    return (z_min, z_max + 1, y_min, y_max + 1, x_min, x_max + 1)


@njit(cache=True, nogil=True)
def _numba_bbox_for_ids_2d(mask, outline, ids_to_delete):
    lookup = np.zeros(65536, dtype=np.bool_)
    for c in ids_to_delete:
        lookup[c] = True

    y_min, x_min = 999999, 999999
    y_max, x_max = -1, -1

    d0, d1 = mask.shape
    for y in range(d0):
        for x in range(d1):
            m = mask[y, x]
            if m != 0 and lookup[m]:
                if y < y_min: y_min = y
                if y > y_max: y_max = y
                if x < x_min: x_min = x
                if x > x_max: x_max = x
                continue

            o = outline[y, x]
            if o != 0 and lookup[o]:
                if y < y_min: y_min = y
                if y > y_max: y_max = y
                if x < x_min: x_min = x
                if x > x_max: x_max = x

    if y_max == -1: return (0, 0, 0, 0)
    return (y_min, y_max + 1, x_min, x_max + 1)

@njit(cache=True, nogil=True)
def _numba_delete_ids_inplace_3d(mask_patch, outline_patch, ids_to_delete):
    lookup = np.zeros(65536, dtype=np.bool_)
    for c in ids_to_delete:
        lookup[c] = True

    d0, d1, d2 = mask_patch.shape
    for z in range(d0):
        for y in range(d1):
            for x in range(d2):
                m = mask_patch[z, y, x]
                if m != 0 and lookup[m]: mask_patch[z, y, x] = 0

                o = outline_patch[z, y, x]
                if o != 0 and lookup[o]: outline_patch[z, y, x] = 0

@njit(cache=True, nogil=True)
def _numba_delete_ids_inplace_2d(mask_patch, outline_patch, ids_to_delete):
    lookup = np.zeros(65536, dtype=np.bool_)
    for c in ids_to_delete:
        lookup[c] = True

    d0, d1 = mask_patch.shape
    for y in range(d0):
        for x in range(d1):
            m = mask_patch[y, x]
            if m != 0 and lookup[m]: mask_patch[y, x] = 0

            o = outline_patch[y, x]
            if o != 0 and lookup[o]: outline_patch[y, x] = 0

@njit(cache=True, nogil=True)
def _numba_get_cell_mean_3d(mask, image, cell_id):
    sum_val = 0.0
    count = 0
    d0, d1, d2 = mask.shape
    for z in range(d0):
        for y in range(d1):
            for x in range(d2):
                if mask[z, y, x] == cell_id:
                    sum_val += image[z, y, x]
                    count += 1
    if count == 0:
        return 0.0
    return sum_val / count

@njit(cache=True, nogil=True)
def _numba_get_cell_mean_2d(mask, image, cell_id):
    sum_val = 0.0
    count = 0
    d0, d1 = mask.shape
    for y in range(d0):
        for x in range(d1):
            if mask[y, x] == cell_id:
                sum_val += image[y, x]
                count += 1
    if count == 0:
        return 0.0
    return sum_val / count

# ==========================================
# NUMBA WARM-UP
# ==========================================

_dummy_uint16_3d = np.zeros((3, 10, 10), dtype=np.uint16)
_dummy_uint16_2d = np.zeros((10, 10), dtype=np.uint16)
_dummy_mask_flat = np.zeros(10, dtype=np.uint16)
_dummy_rgba = np.zeros((10, 10, 4), dtype=np.uint8)
_uint8_3d = np.zeros((3, 10, 10), dtype=np.uint8)
_dummy_ids_to_delete = np.array([1], dtype=np.uint16)
_sliced_uint16_3d = _dummy_uint16_3d[1:-1, 1:-1, 1:-1]
_sliced_uint16_2d = _dummy_uint16_2d[1:-1, 1:-1]

_numba_count(_dummy_uint16_3d)
_numba_count(_dummy_uint16_2d)
_numba_count(_dummy_uint16_3d[0])

_numba_shift_mask(_dummy_mask_flat, _dummy_mask_flat)

_numba_process_2d_slice(_dummy_uint16_2d)
_numba_process_2d_slice(_dummy_uint16_3d[0])

_numba_build_canvas(_dummy_uint16_2d, _dummy_uint16_2d, _dummy_rgba, 0,0,0,0, 0,0,0)

_numba_bbox_for_ids_3d(_dummy_uint16_3d, _dummy_uint16_3d, _dummy_ids_to_delete)
_numba_bbox_for_ids_2d(_dummy_uint16_2d, _dummy_uint16_2d, _dummy_ids_to_delete)
_numba_delete_ids_inplace_3d(_dummy_uint16_3d, _dummy_uint16_3d, _dummy_ids_to_delete)
_numba_delete_ids_inplace_2d(_dummy_uint16_2d, _dummy_uint16_2d, _dummy_ids_to_delete)

_numba_bbox_for_ids_3d(_sliced_uint16_3d, _sliced_uint16_3d, _dummy_ids_to_delete)
_numba_bbox_for_ids_2d(_sliced_uint16_2d, _sliced_uint16_2d, _dummy_ids_to_delete)
_numba_delete_ids_inplace_3d(_sliced_uint16_3d, _sliced_uint16_3d, _dummy_ids_to_delete)
_numba_delete_ids_inplace_2d(_sliced_uint16_2d, _sliced_uint16_2d, _dummy_ids_to_delete)

_numba_get_cell_mean_3d(_dummy_uint16_3d, _dummy_uint16_3d, 1)
_numba_get_cell_mean_2d(_dummy_uint16_2d, _dummy_uint16_2d, 1)

del _dummy_uint16_3d, _dummy_uint16_2d, _dummy_mask_flat, _dummy_rgba