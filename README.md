# SAM2 Points and Box Segmentation

This repository contains a modified version of the official [SAM2](https://github.com/facebookresearch/segment-anything-2) repository. This version enhances the original functionality by adding support for handling both point(s) and box prompts for video segmentation.

## Features

- **Point(s) and Box Prompts**: The script can handle both point(s) and box prompts for segmenting objects in videos.
- **Frame Extraction**: Extract frames from a video using OpenCV.
- **Segmentation Visualization**: Visualize segmentation results on video frames.
- **Video Creation**: Create a video from segmented frames.
- **Device Compatibility**: Supports CUDA, MPS, and CPU for computation.

## Requirements
`python>=3.10` <br />
`torch>=2.3.1` <br />
`torchvision>=0.18.1` <br />

## Installation

To install the required dependencies, run the following command:

```bash
python3 -m pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'
```
Or follow these steps:
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 & pip install -e .
pip install -e ".[demo]"
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```
## Acknowledgements
This project is based on the official SAM2 repository by Facebook Research. Special thanks to the authors for their work on the original implementation
