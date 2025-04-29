from __future__ import annotations

import math
from datetime import datetime
from enum import Enum

class Map:
    def __init__(self, id: int, scale: float, height: int, width: int, walkable_image_bytes: bytes | None = None) -> None:
        self.id: int = id
        self.scale: float = scale
        self.width: int = width
        self.height: int = height
        self.walkable_image_bytes: bytes | None = walkable_image_bytes

    def pixel_is_walkable(self, x: int, y: int) -> bool:
        if self.walkable_image_bytes is not None:
            flatten_index: int = x if y == 0 else x + (self.width * y)
            byte_index, reminder = divmod(flatten_index, 8)
            bit_index: int = 7 - reminder
            bit = (self.walkable_image_bytes[byte_index] >> bit_index) & 1  # Shift digits and remove sign offset
            return bit == 1
        else:
            return True

    def __repr__(self) -> str:
        return self.to_dict().__repr__()

    def to_dict(self) -> dict[str, any]:
        return {
            "id": self.id,
            "scale": self.scale,
            "height": self.height,
            "width": self.width
        }

    @staticmethod
    def from_dict(d: dict) -> Map:
        return Map(int(d["id"]), int(d["scale"]), int(d["height"]), int(d["width"]))