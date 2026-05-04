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

def mask_shifting(mask_data,deleted_mask_id:int,slice_id:int|None = None):
    """
    Shifts the mask when a mask got deleted to restore an order without gaps.

    Args:
        mask_data (np.array): the mask data.
        deleted_mask_id (int): the id of the deleted mask.
        slice_id (int): the id of the slice when the mask is 3d.

    Raises:
          ValueError: if the deleted_mask_id is smaller or equal to 0.
    """
    if deleted_mask_id < 1:
        raise ValueError("deleted_mask_id must be greater than 0")

    mask = mask_data["masks"]
    outline = mask_data["outlines"]

    if mask.ndim == 3:
        if slice_id < 0:
            raise ValueError("slice_id should be non-negative")
        mask_slice = np.take(mask, slice_id, axis=0).astype(np.uint16)
        mask_slice[mask_slice>deleted_mask_id] -= 1
        mask[slice_id, :, :] = mask_slice
    else:
        mask[mask > deleted_mask_id] -= 1

    if outline.ndim == 3:
        if slice_id < 0:
            raise ValueError("slice_id should be non-negative")
        outline_slice = np.take(outline, slice_id, axis=0).astype(np.uint16)
        outline_slice[outline_slice>deleted_mask_id] -= 1
        outline[slice_id, :, :] = outline_slice
    else:
        outline[outline>deleted_mask_id] -= 1

def search_free_id(mask,outline):
    """
    Search in a NumPy array of integers (e.g., [1,1,2,2,3,4,5,5,7,7]) for the first missing number (in this case, 6).
    If no gap is found, return the highest value + 1.
    """

    max_val = max(mask.max(), outline.max())
    if max_val == 0:
        return 1

    counts = np.bincount(mask.ravel(), minlength=max_val + 1)
    counts += np.bincount(outline.ravel(), minlength=max_val + 1)

    zero_indices = np.where(counts[1:] == 0)[0]

    if zero_indices.size > 0:
        return int(zero_indices[0] + 1)
    else:
        return int(max_val + 1)