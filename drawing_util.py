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

def mask_shifting(mask_data):
    """
    Shifts the mask when a mask got deleted to restore an order without gaps.

    Args:
        mask_data (np.array): the mask data.
        deleted_mask_id (int): the id of the deleted mask.
        slice_id (int): the id of the slice when the mask is 3d.

    Raises:
          ValueError: if the deleted_mask_id is smaller or equal to 0.
    """
    mask = mask_data["masks"]
    outline = mask_data["outlines"]

    target_mask = mask
    target_outline = outline

    all_ids = np.unique(np.concatenate([np.unique(target_mask), np.unique(target_outline)]))
    all_ids = all_ids[all_ids != 0]

    if len(all_ids) == 0 or np.array_equal(all_ids, np.arange(1, len(all_ids) + 1)):
        return

    max_id = all_ids[-1]
    lookup = np.arange(max_id + 1, dtype=np.uint16)

    for new_id, old_id in enumerate(all_ids, 1):
        lookup[old_id] = new_id

    mask_data["masks"] = lookup[mask]
    mask_data["outlines"] = lookup[outline]

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