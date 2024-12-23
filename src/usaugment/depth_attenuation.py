"""
An implementation of the depth attenuation transform described in "Myocardial Function
Imaging in Echocardiography Using Deep Learning" (Ostvik et al., 2021).
"""

import random
from typing import Any, Dict, Tuple

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class DepthAttenuation(ImageOnlyTransform):
    """
    An implementation of the depth attenuation transform described in "Myocardial
    Function Imaging in Echocardiography Using Deep Learning" (Ostvik et al., 2021).
    """

    def __init__(
        self,
        attenuation_rate: float | Tuple[float, float] = (0.0, 3.0),
        max_attenuation: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super(DepthAttenuation, self).__init__(p=p)
        self.attenuation_rate = attenuation_rate
        self.max_attenuation = max_attenuation

    def apply(self, img: np.ndarray, **params: Any):
        img = img.copy()

        attenuation_map = self._generate_attenuation_map(
            *img.shape[:2], scan_mask=params["scan_mask"]
        ).astype(img.dtype)

        scan_mask = params["scan_mask"].astype(bool)
        attenuation_map = np.where(scan_mask, attenuation_map, 1.0)

        if img.ndim == 2:
            # Single-channel image
            img = img * attenuation_map
        else:
            # Multi-channel image
            img = img * attenuation_map[:, :, None]

        return img

    def get_params_dependent_on_data(
        self, params: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"scan_mask": data["scan_mask"]}

    def _generate_attenuation_map(self, height, width, scan_mask):
        """Generate an attenuation map for the image."""
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xv, yv = np.meshgrid(x, y)
        distances = np.sqrt((xv - 0.5) ** 2 + yv**2)

        if isinstance(self.attenuation_rate, tuple) or isinstance(
            self.attenuation_rate, list
        ):
            attenuation_rate = random.uniform(*self.attenuation_rate)
        else:
            attenuation_rate = self.attenuation_rate

        attenuation_map = self._bounded_exponential_decay(
            distances, attenuation_rate, self.max_attenuation
        )
        attenuation_map = attenuation_map * scan_mask

        return attenuation_map

    def _bounded_exponential_decay(
        self, distances, attenuation_rate, max_attenuation=0
    ):
        """Calculate the intensity of the beam after a given distance using a bounded
        exponential decay.
        """
        intensities = (1 - max_attenuation) * np.exp(
            -attenuation_rate * distances
        ) + max_attenuation
        return intensities
