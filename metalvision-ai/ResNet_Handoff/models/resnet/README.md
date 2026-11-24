# METALVISION AI — ResNet-50 Defect Classifier

## Status: TRAINED AND CHECKPOINT AVAILABLE

The ResNet-50 model is trained and integrated into the standalone ResNet inference module.

### Model details

- Architecture: ResNet-50
- Framework: PyTorch
- Input size: 224 x 224
- Input channels: 3 (RGB)
- Preprocessing: Aspect-preserving resize, grey padding and ImageNet normalization
- Weights: ImageNet-pretrained ResNet-50, fine-tuned for five defect classes

### Supported defect classes

1. silk_spot
2. inclusion
3. waist_folding
4. rolled_pit
5. water_spot

### Model checkpoint

File: `resnet50_best.pth`

Location: `models/resnet/resnet50_best.pth`

The checkpoint contains the model state dictionary, architecture, class names, image size, normalization parameters, best validation macro-F1 and training epoch.

### Official test evaluation

Evaluated on the original group-aware test split:

- Test samples: 202
- Correct predictions: 193
- Accuracy: 95.54%
- Macro-F1: 0.8606
- Best checkpoint epoch: 15
- Best validation macro-F1: 0.8297

These metrics describe the five-class crop classification task, not full-image defect localization.

### Inference module

Implementation: `backend/resnet_inference.py`

Main functionality:
- Checkpoint loading
- OpenCV BGR crop preprocessing
- Five-class prediction
- Class probability output
- Grad-CAM evidence map generation

Grad-CAM is an evidence visualization, not a segmentation mask or exact defect boundary.

The model expects a defect crop in BGR format. It does not classify a full grayscale X-ray image directly.

### Evaluation reports

Saved in `reports/`:

- `resnet_original_test_evaluation_20261124.txt`
- `resnet_original_test_evaluation_20261124.json`
- `resnet_original_test_evaluation_20261124_class_metrics.csv`
- `resnet_original_test_evaluation_20261124_confusion_matrix.csv`

### Important notes

- The model supports only the five listed defect classes.
- It does not include a normal/defect-free class.
- Predictions are not calibrated probabilities.
- Performance on classes with very few test samples should be interpreted cautiously.
- `backend/resnet_training.py` may represent an older training workflow and should not be assumed to reproduce this checkpoint without verification.

