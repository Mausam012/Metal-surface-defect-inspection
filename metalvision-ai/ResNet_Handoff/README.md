# METALVISION AI - Standalone ResNet Module

## Contents
- backend: ResNet inference and supporting modules
- models/resnet: Trained ResNet-50 checkpoint
- reports: Original test-set evaluation reports

## Requirements
Python 3.12 recommended.

Install dependencies:
pip install -r requirements.txt

## Model
Architecture: ResNet-50
Input: OpenCV BGR defect crop
Input size: 224 x 224
Classes: silk_spot, inclusion, waist_folding, rolled_pit, water_spot

## Basic usage

Run from the package root:

```python
from pathlib import Path
import cv2
from backend.resnet_inference import ResNetModel

model = ResNetModel(Path("models"))
model.load(Path("models/resnet/resnet50_best.pth"))

image = cv2.imread("sample_crop.jpg")
result = model.predict_crop(image)

print(result)
