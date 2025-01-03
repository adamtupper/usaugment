"""An implementation of the Gaussian shadow transform described in "Highlighting nerves
and blood vessels for ultrasound-guided axillary nerve block procedures using neural
networks" (Smistad et al., 2018).
"""

from typing import Any, Dict, Tuple

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


class GaussianShadow(ImageOnlyTransform):
    """
    An implementation of the Gaussian shadow transform described in "Highlighting nerves
    and blood vessels for ultrasound-guided axillary nerve block procedures using neural
    networks" (Smistad et al., 2018).
    """

    def __init__(
        self,
        strength: float | Tuple[float, float] = (0.25, 0.8),
        sigma_x: float | Tuple[float, float] = (0.01, 0.2),
        sigma_y: float | Tuple[float, float] = (0.01, 0.2),
        p: float = 0.5,
    ) -> None:
        super(GaussianShadow, self).__init__(p=p)
        self.strength = strength
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def apply(self, img: np.ndarray, **params: Any):
        img = img.copy()

        shadow_image = self._generate_shadow_image(
            height=img.shape[0], width=img.shape[1], scan_mask=params["scan_mask"]
        ).astype(img.dtype)

        scan_mask = params["scan_mask"].astype(bool)
        shadow_image = np.where(scan_mask, shadow_image, 1.0)

        if img.ndim == 2:
            # Single-channel image
            img = img * shadow_image
        else:
            # Multi-channel image
            img = img * shadow_image[:, :, None]

        return img

    def get_params_dependent_on_data(
        self, params: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {"scan_mask": data["scan_mask"]}

    def _generate_shadow_image(
        self,
        height,
        width,
        scan_mask,
    ):
        """Generate a shadow image."""
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)

        mu_x = np.random.choice(x)
        mu_y = np.random.choice(y)

        if isinstance(self.strength, tuple) or isinstance(self.strength, list):
            strength = np.random.uniform(*self.strength)
        else:
            strength = self.strength

        if isinstance(self.sigma_x, tuple) or isinstance(self.sigma_x, list):
            sigma_x = np.random.uniform(*self.sigma_x)
        else:
            sigma_x = self.sigma_x

        if isinstance(self.sigma_y, tuple) or isinstance(self.sigma_y, list):
            sigma_y = np.random.uniform(*self.sigma_y)
        else:
            sigma_y = self.sigma_y

        sigma_x = sigma_x * width
        sigma_y = sigma_y * height

        shadow_image = 1 - strength * np.exp(
            -((xv - mu_x) ** 2 / (2 * sigma_x**2) + (yv - mu_y) ** 2 / (2 * sigma_y**2))
        )
        shadow_image = shadow_image * scan_mask

        return shadow_image
