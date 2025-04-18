# USAugment

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/adamtupper/usaugment/HEAD?urlpath=%2Fdoc%2Ftree%2Fexamples%2Fostvik_augmentations.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2501.13193-b31b1b.svg)](http://arxiv.org/abs/2501.13193)

![Examples of each augmentation.](figures/readme_banner.png)

USAugment provides ultrasound-specific image transforms for training deep neural networks. It accompanies our article [Revisiting Data Augmentation for Ultrasound Images](http://arxiv.org/abs/2501.13193).

Checkout the [Wiki](https://github.com/adamtupper/usaugment/wiki) to find the documentation, usage examples, and contributing guidelines.

## How to install USAugment

This package can be installed from [PyPI](https://pypi.org/project/usaugment/) by running:

```bash
pip install usaugment
```

## Get started using USAugment

Here's a quick code demo showing how to compose a multiple augmentations together using Albumentations. Note that the augmentations also require a binary scan mask identifying the region of the scan in the image (1 = pixels in the scan, 0 = otherwise).

```python
>>> import albumentations as A
>>> from usaugment.albumentations import DepthAttenuation, GaussianShadow, HazeArtifact, SpeckleReduction
>>> transform = A.Compose(
...     [
...         DepthAttenuation(p=0.5),
...         GaussianShadow(p=0.5),
...         HazeArtifact(p=0.5),
...         SpeckleReduction(p=0.5),
...     ],
...     additional_targets={"scan_mask": "mask"}
...)
>>> image = ... # Load image
>>> scan_mask = ... # Load scan mask
>>> transformed = transform(image=image, scan_mask=scan_mask)
```

Checkout the [Documentation](https://github.com/adamtupper/usaugment/wiki) for more detailed examples and information about scan masks.

## Running the examples

There's a notebook in the examples directory showing how to use and visualize the effect of each augmentation. You can run these examples using Binder (see the tag at the top of page) or locally by installing the optional `examples` dependencies (Matplotlib and Jupyter Notebook):

```bash
pip install '.[examples]'
```

## Contributing

I'd love for this package to grow and flourish into a resource that anyone with an interest in training deep neural networks for ultrasound analysis tasks can pick up and use quickly and easily. Any help addressing bugs, contributing new augmentations, or any other improvements are welcome and appreciated! I only ask that you respect the community guidelines laid out in the `CODE_OF_CONDUCT.md`. For more information on how to contribute, checkout out the [Documentation](https://github.com/adamtupper/usaugment/wiki).

To ensure that your code meets the style guidelines etc., make sure you install the optional development dependencies:

```bash
pip install '.[dev]'
nbstripout --install
pre-commit install
```

## How to cite USAugment

If you use the augmentations in USAugment in your research, please cite our article [Revisiting Data Augmentation for Ultrasound Images](http://arxiv.org/abs/2501.13193). This helps more people find and use the package and encourages us to continue maintaining and improving it!

```
@misc{tupper2025,
      title={Revisiting Data Augmentation for Ultrasound Images}, 
      author={Adam Tupper and Christian Gagné},
      year={2025},
      eprint={2501.13193},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2501.13193}, 
}
```