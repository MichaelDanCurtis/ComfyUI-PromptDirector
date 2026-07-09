from __future__ import annotations

import base64
from io import BytesIO
from typing import Any


def image_to_png_data_uri(image: Any) -> str:
    try:
        import numpy as np
        from PIL import Image
        import torch
    except Exception as exc:
        raise RuntimeError("PIL, numpy, and torch are required to encode image inputs") from exc

    tensor = image
    if torch.is_tensor(tensor):
        array = tensor.detach().cpu().float().numpy()
    else:
        array = np.asarray(tensor)

    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 or 4 dimensions, got shape {array.shape}")

    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)

    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype("uint8")
    if array.shape[-1] == 1:
        array = array[:, :, 0]

    image_obj = Image.fromarray(array)
    buffer = BytesIO()
    image_obj.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
