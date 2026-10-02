"""
image.py — Opening an image from whatever the caller has, for the image modules' load_image().
"""

import io

import numpy as np
from PIL import Image


def open_image(source) -> Image.Image:
    """A path, raw bytes, a file-like object (e.g. a Streamlit upload), a PIL image or a uint8 array."""
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, np.ndarray):
        return Image.fromarray(source)
    if isinstance(source, (bytes, bytearray)):
        return Image.open(io.BytesIO(source))
    return Image.open(source)
