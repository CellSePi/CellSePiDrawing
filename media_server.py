import socket
import threading
import uvicorn
from fastapi import FastAPI, Response
from collections import OrderedDict
import cv2


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class RenderedBytesCache:
    def __init__(self, max_items=100):
        self.cache = OrderedDict()
        self._max_items = max_items
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def add(self, key, value, shape, is_3d):
        with self.lock:
            self.cache[key] = {
                "bytes": value,
                "shape": shape,
                "is_3d": is_3d
            }
            self.cache.move_to_end(key)
            if len(self.cache) > self._max_items:
                self.cache.popitem(last=False)

    def has_key(self, key):
        with self.lock:
            return key in self.cache

    def clear(self):
        with self.lock:
            self.cache.clear()


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
        self._keys = None
        self._shape = None
        self._is_3d = False
        self._rendered_cache = RenderedBytesCache()

        self.app.get("/image")(self._get_image_endpoint)
        self._initialized = True
        self.start()

    def _get_image_endpoint(self):
        with self._image_lock:
            image = self._image
            keys = self._keys
            shape = self._shape
            is_3d = self._is_3d

        cached_bytes = self._rendered_cache.get(keys)
        if cached_bytes is not None:
            return Response(content=cached_bytes["bytes"], media_type="image/png")

        if image is None:
            return Response(status_code=404)

        _, buffer = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        bytes_data = buffer.tobytes()
        self._rendered_cache.add(keys, bytes_data,shape,is_3d)
        return Response(content=bytes_data, media_type="image/png")

    def update_image(self, image, keys, shape, is_3d):
        with self._image_lock:
            self._image = image
            self._keys = keys
            self._shape = shape
            self._is_3d = is_3d

    def get_cached_entry(self, keys):
        return self._rendered_cache.get(keys)

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