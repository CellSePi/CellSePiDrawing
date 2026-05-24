import socket
import threading

import numpy as np
import uvicorn
from fastapi import FastAPI, Response
import cv2


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def convert_npy_to_canvas(mask, outline, mask_color, outline_color, opacity, slice_id=-1):
    """
    handles the conversion of the given file data

    Args:
        mask= the mask data stored in the numpy directory
        outline= the outline data stored in the numpy directory
    """
    if mask.ndim == 3:  # if 3d get the given slice or get a max projection
        mask_slice = mask[slice_id] if slice_id >= 0 else mask.any(axis=0)
        outline_slice = outline[slice_id] if slice_id >= 0 else outline.any(axis=0)
    else:  # 2d nothing to do
        mask_slice = mask
        outline_slice = outline

    # filter values greater than 0
    mask_bool = mask_slice > 0
    outline_bool = outline_slice > 0

    has_mask = mask_bool.any()
    has_outline = outline_bool.any()

    image_mask = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 4),
                          dtype=np.uint8)  # uint8 because here we dont use the cell_id's

    if has_mask:
        image_mask[mask_bool] = (mask_color[2], mask_color[1], mask_color[0], opacity)

    if has_outline:
        image_mask[outline_bool] = (outline_color[2], outline_color[1], outline_color[0], 255)

    return image_mask



class MediaServer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MediaServer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, host="127.0.0.1"):
        if self._initialized: return
        self._image_lock = threading.Lock()
        self._mask_lock = threading.Lock()
        self.app = FastAPI()
        self.host = host
        self.port = 0
        self._server_thread = None
        self._image = None
        self._mask_update_data = None

        self.app.get("/image")(self._get_image_endpoint)
        self.app.get("/mask")(self._get_mask_endpoint)
        self._initialized = True
        self.start()

    def _get_image_endpoint(self):
        with self._image_lock:
            if self._image is None:
                return Response(status_code=404)
            _, buffer = cv2.imencode('.png', self._image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return Response(content=buffer.tobytes(), media_type="image/png")

    def _get_mask_endpoint(self):
        with self._mask_lock:
            if self._mask_update_data is None:
                return Response(status_code=404)
            data = self._mask_update_data

        masks, outlines, mask_color, outline_color, opacity, slice_id = data
        image_mask = convert_npy_to_canvas(masks, outlines, mask_color, outline_color, opacity, slice_id)
        _, buffer = cv2.imencode('.webp', image_mask, [cv2.IMWRITE_WEBP_QUALITY, 101])
        return Response(content=buffer.tobytes(), media_type="image/webp")

    def update_image(self, image):
        with self._image_lock:
            self._image = image

    def update_mask(self, mask_update_data):
        with self._mask_lock:
            self._mask_update_data = mask_update_data

    def start(self):
        self.port = _find_free_port()
        self._server_thread = threading.Thread(
            target=lambda: uvicorn.run(self.app, host=self.host, port=self.port, log_level="error"),
            daemon=True
        )
        self._server_thread.start()

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"