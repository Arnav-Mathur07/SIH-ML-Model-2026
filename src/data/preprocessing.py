import cv2
import numpy as np
from src.utils.config import Config

class SonarPreprocessor:
    def __init__(self, config: Config):
        self.config = config.preprocessing

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if not self.config.enabled:
            return self._ensure_3channel(image)

        img = image.copy()
        
        # Convert to grayscale if it's RGB but actually grayscale data
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        if self.config.median_filter:
            img = cv2.medianBlur(img, self.config.median_kernel)

        if self.config.bilateral_filter:
            img = cv2.bilateralFilter(
                img, 
                self.config.bilateral_d, 
                self.config.bilateral_sigma_color, 
                self.config.bilateral_sigma_space
            )

        if self.config.clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit, 
                tileGridSize=tuple(self.config.clahe_tile_grid)
            )
            img = clahe.apply(img)

        if self.config.normalize:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        return self._ensure_3channel(img)

    def preprocess_with_comparison(self, image: np.ndarray) -> dict:
        results = {"Original": image.copy()}
        
        if not self.config.enabled:
            return results

        img = image.copy()
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.config.median_filter:
            img = cv2.medianBlur(img, self.config.median_kernel)
            results["Median Filter"] = self._ensure_3channel(img)

        if self.config.bilateral_filter:
            img = cv2.bilateralFilter(
                img, 
                self.config.bilateral_d, 
                self.config.bilateral_sigma_color, 
                self.config.bilateral_sigma_space
            )
            results["Bilateral Filter"] = self._ensure_3channel(img)

        if self.config.clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit, 
                tileGridSize=tuple(self.config.clahe_tile_grid)
            )
            img = clahe.apply(img)
            results["CLAHE"] = self._ensure_3channel(img)

        if self.config.normalize:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            results["Normalized (Final)"] = self._ensure_3channel(img)

        return results

    def _ensure_3channel(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

class TilingEngine:
    def __init__(self, tile_size: int, overlap: float):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1.0 - overlap))

    def split(self, image: np.ndarray) -> list:
        tiles = []
        h, w = image.shape[:2]
        
        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                
                # Adjust if tile is smaller than tile_size at edges
                y_start = max(0, y_end - self.tile_size)
                x_start = max(0, x_end - self.tile_size)
                
                tile = image[y_start:y_end, x_start:x_end]
                tiles.append({
                    "image": tile,
                    "x_offset": x_start,
                    "y_offset": y_start,
                    "row": y // self.stride,
                    "col": x // self.stride
                })
        return tiles
