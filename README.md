<h1 align='center'> DRAFTS </h1>

<div align="center">

_✨ Deep learning-based RAdio Fast Transient Search pipeline✨_

<img src="https://counter.seku.su/cmoe?name=APOD&theme=r34" /><br>

</div>

<p align="center">
  <a href="https://github.com/SukiYume/DRAFTS">
    <img src="https://img.shields.io/badge/TransientSearch-DRAFTS-da282a" alt="release">
  </a>
  <a href="https://github.com/SukiYume/DRAFTS/stargazers">
    <img src="https://img.shields.io/github/stars/SukiYume/DRAFTS.svg?label=Stars&logo=github" alt="release">
  </a>
</p>


## <div align="center">Description</div>

> Traditional single-pulse search techniques, such as Presto and Hemidall, suffer a series of challenges, including intricate installation procedures, sluggish execution, incomplete outcomes, and a reliance on human result verification.

> We devised a Deep learning-based RAdio Fast Transient Search pipeline (DRAFTS) to address the aforementioned concerns. This pipeline contains: a. CUDA accelerated de-dispersion, b. Object detection model extracts TOA and DM of FRB signal, c. The binary classification model checks the authenticity of the candidate signal.

> All the code implementation is written with Python, thereby facilitating straightforward installation on any operating system. Testing on real data from FAST, real-time searching can be achieved on an RTX 2070S graphics card. Compared to Hemidall, it almost doubled the detection number of bursts. The classification model achieves an accuracy exceeding 99% on FAST and GBT data, thereby significantly reducing the reliance on manual inspection for search results.


## <div align="center">Usage</div>

The training code for our object detection and classification models is contained in the `BinaryClass` and `ObjectDet` folders. Training data and pre-trained models are available on [Hugging Face](https://huggingface.co/TorchLight/DRAFTS).

The required packages are listed in the `requirements.txt` file.


### Object Detection

For the object detection model, download the data into the `ObjectDet/Data` folder. Place the label file `data_label.txt` in the same directory as `centernet_train.py`. Execute the following command to start training the CenterNet model with ResNet18 as the backbone. To train the model with ResNet50 as the backbone, replace `resnet18` with `resnet50`.

```bash
python centernet_train.py resnet18
```


### Binary Classification

For the classification model, download the data into the `BinaryClass/Data` folder. The data should be divided into `True` and `False` subfolders within this directory. Execute the following command to start training the classification model with ResNet18 as the backbone. To train the model with ResNet50 as the backbone, replace `resnet18` with `resnet50`.

```bash
python binary_train.py resnet18 BinaryNet
```

In `binary_model.py`, we also constructed a classification model that supports arbitrary image sizes using `SpatialPyramidPool2D`. To train this model, execute the following command

```bash
python binary_train.py resnet18 SPPResNet
```


### Check Performance

To evaluate the performance of the models, we use the [FAST-FREX](https://doi.org/10.57760/sciencedb.15070) independent dataset. Place all `FITS` files from FAST-FREX into the `CheckRes/RawData/Data` folder. Place the trained model checkpoints in the same directory as the Python files.

Execute the respective scripts to evaluate the performance of the classification models (files with ddmt in their names) and object detection models (files with cent in their names) under ResNet18/50.

Note that the classification model depends on the `binary_model.py` file, and the object detection model depends on the `centernet_utils.py` and `centernet_model.py` files.


## <div align="center">Installation</div>

To install the required packages, run

```bash
pip install -r requirements.txt
```


## <div align="center">Other</div>

Welcome contributions!

