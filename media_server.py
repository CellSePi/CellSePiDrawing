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
        self.app = FastAPI()
        self.host = host
        self.port = 0
        self._server_thread = None
        self._image = None
        self._mask_update_data = None

        self.app.get("/image")(self._get_image_endpoint)
        self._initialized = True
        self.start()

    def _get_image_endpoint(self):
        with self._image_lock:
            if self._image is None:
                return Response(status_code=404)
            _, buffer = cv2.imencode('.png', self._image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        return Response(content=buffer.tobytes(), media_type="image/png")

    def update_image(self, image):
        with self._image_lock:
            self._image = image

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