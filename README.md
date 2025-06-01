<p align="center">
    <img src="resources/ai-frame-logo.png" width="256" /><br>
    <strong>A lightweight framework for Neural Networks and AI</strong><br><br>
    <img src="https://img.shields.io/badge/Python-v3.13-red?logo=python&logoColor=3776AB&labelColor=3e474f" />
    <img src="https://img.shields.io/badge/Numpy-v2.2.5-blue?logo=numpy&labelColor=013243" />
    <img src="https://img.shields.io/badge/OpenCL-v2025.1-greenlabelColor=013243" />
</p>


## 1. 📖 About

AIFrame aims to be a **lightweight framework** written in pure Python and accelerated by **OpenCL** and **Numpy**.
This repository is a clean and more well-written version than the actual prototype that isn't available for public use.

**Currently, the project is under development and will need some time to be actually used!**

## 2. 🔧 Features (planned)

- ✨ Simplicity for fast prototyping
- 🏎️ Speed and performance for low-end hardware (CPU & GPU support)
- 💻 Training pipelines for easy training
- 📄 Buildscripts for automated use
- 🔌 Low-level access for custom logic
- 💾 Custom file format
- 🗺️ Adaption for other file formats

## 3. Demo Setup
Current demo (`demo-01.06.2025.py`) needs following requirements:
`gzip, requests, numpy`
(no specific version, newest is enough).

Run the demo.
```shell
python demo-01.06.2025.py
```
This demo downloads the MNIST-Dataset from `https://storage.googleapis.com/cvdf-datasets/mnist/`. After an example training of all 60k samples, it will evaluate the 10k test samples.
Results may very in speed and result, but training accuracy should be around `94-98%`.
Tested on a `AMD Ryzen 5 3600 (6 Cores)`; training took `12.5 seconds` to complete.