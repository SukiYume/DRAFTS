<div align="center">

#  DRAFTS

_✨ Deep learning-based RAdio Fast Transient Search pipeline (Under Construction)✨_

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


## <div align="center">The paper is in preparation.</div>

<details open>
<summary>Binary Classification Model</summary>
</br>
<div align="left">
The binary classification model not only works in conjunction with the object detection model, but it can also be employed independently for burst searching when we already knew the dispersion information of a transient source.

Currently, the code we have uploaded enables the use of this binary classification model to search for FRBs with known dispersion information from the FAST data. The code for training the model will be gradually uploaded as the project progresses. The models have been hosted on the Hugging Face project page for [DRAFTS](https://huggingface.co/TorchLight/DRAFTS).
</div>
</details>

<details open>
<summary>Object Detection Model</summary>
</br>
<div align="center">Waiting for Update</div>
</details>

<details open>
<summary>PyTorch inplementation</summary>
</br>
<div align="center">
Stay tuned for updates.</div>
<div align="left">
In the current version of the code, the deep learning model is implemented using the TensorFlow/Keras framework. An upcoming PyTorch version is nearing completion, featuring a slightly different network structure but enhanced functionality. One notable improvement is that while the TensorFlow version of the binary classification model only supports input images of size 512x512, the PyTorch version allows for input images of any size.</div>
</details>