<h1 align='center'> DRAFTS </h1>

<div align="center">

✨ **Deep learning-based RAdio Fast Transient Search pipeline** ✨

[![TransientSearch](https://img.shields.io/badge/TransientSearch-DRAFTS-da282a)](https://github.com/SukiYume/DRAFTS)
[![GitHub Stars](https://img.shields.io/github/stars/SukiYume/DRAFTS.svg?label=Stars&logo=github)](https://github.com/SukiYume/DRAFTS/stargazers)
[![arXiv](https://img.shields.io/badge/arXiv-2410.03200-b31b1b.svg)](https://arxiv.org/abs/2410.03200)
[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)

[Description](#description) • 
[Installation](#installation) • 
[Usage](#usage) • 
[Models](#models) • 
[Performance](#performance) • 
[Real Data Search](#search-in-real-observation-data) • 
[Contributing](#contributing)

</div>

![DRAFTS WorkFlow](./WorkFlow.png)

## Description

**DRAFTS** is a Deep learning-based RAdio Fast Transient Search pipeline designed to address limitations in traditional single-pulse search techniques like Presto and Heimdall.

Traditional methods often face challenges including:
- Complex installation procedures
- Slow execution speeds
- Incomplete search results
- Heavy reliance on manual verification

Our pipeline offers three key components:
1. **CUDA-accelerated de-dispersion** for faster processing
2. **Object detection model** to extract Time of Arrival (TOA) and Dispersion Measure (DM) of FRB signals
3. **Binary classification model** to verify candidate signal authenticity

**Key advantages:**
- Written entirely in Python for easy cross-platform installation
- Achieves real-time searching on consumer GPUs (tested on RTX 2070S)
- Nearly doubles burst detection compared to Heimdall
- Classification accuracy exceeding 99% on FAST and GBT data
- Significantly reduces manual verification requirements

📄 **Publication:** [DRAFTS: A Deep Learning-Based Radio Fast Transient Search Pipeline (arXiv:2410.03200)](https://arxiv.org/abs/2410.03200)

## Installation

Install all required dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Training data and pre-trained models are available on HuggingFace:
- [DRAFTS-Data](https://huggingface.co/datasets/TorchLight/DRAFTS)
- [DRAFTS-Model](https://huggingface.co/TorchLight/DRAFTS)

### Models

#### Object Detection

The object detection training code is in the `ObjectDet` folder.

1. Download data to `ObjectDet/Data`
2. Place `data_label.txt` in the same directory as `centernet_train.py`
3. Train using:

```bash
python centernet_train.py resnet18  # Use 'resnet50' for ResNet50 backbone
```

#### Binary Classification

The classification training code is in the `BinaryClass` folder.

1. Download data to `BinaryClass/Data`
2. Ensure data is organized in `True` and `False` subfolders
3. Train standard model:

```bash
python binary_train.py resnet18 BinaryNet  # Use 'resnet50' for ResNet50 backbone
```

4. Train with arbitrary image size support using `SpatialPyramidPool2D`:

```bash
python binary_train.py resnet18 SPPResNet
```

### Performance

To evaluate model performance:

1. Use the [FAST-FREX](https://doi.org/10.57760/sciencedb.15070) independent dataset
2. Place FITS files in `CheckRes/RawData/Data`
3. Place trained model checkpoints in the same directory as the Python files
4. Run evaluation scripts:
   - Files with `ddmt` for classification models
   - Files with `cent` for object detection models

**Dependencies:**
- Classification models depend on `binary_model.py`
- Object detection models depend on `centernet_utils.py` and `centernet_model.py`

## Search in Real Observation Data

For complete FAST observation data:

1. Refer to `d-center-main.py` and `d-resnet-main.py`
2. Modify the `data path` and `save path`
3. Run the file

**Note:** The current search program automatically adapts to FAST and GBT observation data. For other telescopes, modify the `load_fits_file` function and related data reading functions.

## Contributing

Contributions to DRAFTS are welcome! Please feel free to submit issues or pull requests.

---

<div align="center">
  <sub>Searching for cosmic signals 🔭✨📡</sub>
</div>