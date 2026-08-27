# Comprehensive Guide to Vision Transformer (ViT) Image Classification Pipeline

## Table of Contents

- [Introduction](#introduction)
- [Author Information](#author-information)
- [Conceptual Foundations](#conceptual-foundations)
- [Environment Setup](#environment-setup)
- [Dataset Organization](#dataset-organization)
- [Configuration Guide](#configuration-guide)
- [Execution Instructions](#execution-instructions)
- [Experimental Results](#experimental-results)
- [Advanced Features & Architectures](#advanced-features--architectures)
- [Output Structure & Interpretation](#output-structure--interpretation)
- [Example Console Output](#example-console-output)
- [Explainability & XAI Features](#explainability--xai-features)
- [Appendix: Inference Example](#appendix-inference-example)

---

## Introduction

This repository provides a robust, production-grade PyTorch pipeline for image classification utilizing Vision
Transformers (ViT) and other state-of-the-art architectures. It is specifically designed to accommodate students and
researchers, offering an accessible yet highly configurable framework. The pipeline supports automatic dataset
splitting, advanced data preprocessing (including histopathology-specific augmentations), multiple cutting-edge models,
modern optimizers, comprehensive evaluation metrics, and integrated Explainable AI (XAI) capabilities.

## Author Information

- **Author:** Hossam Magdy Balaha
- **Email:** hmbala01@louisville.edu
- **Online CV:** [https://hossambalaha.github.io/](https://hossambalaha.github.io/)

## Conceptual Foundations

Prior to examining the source code, it is advantageous to comprehend the underlying computational processes. The
following video resources are recommended to establish a foundational understanding.

### General Deep Learning Fundamentals

1. [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk)
2. [A Gentle Introduction to Machine Learning](https://www.youtube.com/watch?v=Gv9_4yMHFhI)

### PyTorch Fundamentals

1. [PyTorch Crash Course: Deep Learning in Python](https://www.youtube.com/watch?v=uq7sbUlIDR8)
2. [PyTorch in 1 Hour](https://www.youtube.com/watch?v=r1bquDz5GGA)

### Vision Transformers (ViT) Track

1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://www.youtube.com/watch?v=TrdevFK_am4)
2. [Vision Transformer Quick Guide - Theory and Code](https://www.youtube.com/watch?v=j3VNqtJUoz0)
3. [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://www.youtube.com/watch?v=tFYxJZBAbE8)
4. [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://www.youtube.com/watch?v=eaZt9asVYH0)

### Convolutional Neural Network (CNN) Foundations

1. [PyTorch CNN Part 1](https://www.youtube.com/watch?v=g6eHItPHd7k)
2. [PyTorch CNN Part 2](https://www.youtube.com/watch?v=_QpiHNykR9k)

## Environment Setup

### Step 1: Install Python

Navigate to [Python.org](https://www.python.org/downloads/) and download Python 3.9 or newer.  
**Critical for Windows Users:** Ensure the checkbox labeled "Add Python to PATH" is selected during installation.  
**Alternative:** Download [Anaconda](https://www.anaconda.com/download) for a pre-configured scientific environment.

### Step 2: Install Required Libraries

Open your Command Prompt (Windows) or Terminal (Mac/Linux) and execute the following commands:

```bash
# Install the core required libraries.
pip install numpy seaborn matplotlib scikit-learn pillow pyyaml split-folders tqdm torch torchvision torchaudio timm ftfy regex

# Install OpenAI CLIP directly from the official repository.
pip install git+https://github.com/openai/CLIP.git

# Install state-of-the-art optimizers for advanced Vision Transformer training.
pip install lion-pytorch prodigyopt schedulefree

# Install the HMB Helpers Package for Explainable AI (XAI) and advanced utility functions.
pip install "hmb-helpers[cv,pytorch]"
```

*(Note: For GPU acceleration, visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) to obtain
the correct CUDA command. A CUDA-enabled GPU is strongly recommended for Vision Transformers.)*

## Dataset Organization

The pipeline automatically manages dataset partitioning. Manual creation of training, validation, and testing
directories is unnecessary.

Create a root directory (e.g., `Dataset`) and place your images into class-specific subdirectories:

```text
Dataset/
├── ClassA/          <-- Folder name becomes the class name.
│   ├── img1.jpg
│   └── img2.png
├── ClassB/          <-- Another class.
│   ├── img3.jpg
│   └── img4.png
└── ClassC/          <-- Multiple classes are supported.
    └── img5.jpg
```

Upon execution, the code detects the absence of `train`, `val`, and `test` directories and automatically partitions the
data (80% training, 10% validation, 10% testing) using the `split-folders` library. If the splits already exist, they
are utilized directly.

## Configuration Guide

The pipeline utilizes a YAML configuration file to manage parameters, facilitating reproducible experiments and grid
searches. Create a file named `config.pytorch.yaml` in your working directory.

### Example Configuration File

```yaml
# ==============================================================================
# VIT IMAGE CLASSIFICATION PIPELINE - CONFIGURATION FILE
# ==============================================================================

# --- Data and Output Paths ---
DataDir: "/path/to/your/dataset"
OutputDir: "/path/to/your/output"

# --- Experiment Grid ---
ModelName:
  - "SwinTransformerV2"
  - "EVA02"

Optimizer:
  - "Prodigy"

LossFunction:
  - "Focal"

# --- Training Hyperparameters ---
ImageSize: 224
BatchSize: 64
LearningRate: 0.0001
NumEpochs: 500
UseAugmentation: true
Device: "cuda"
Patience: 25

# --- Advanced Training Techniques ---
UseAmp: true
AccumulationSteps: 1
UseEma: true
UseMixup: false
MixupAlpha: 0.2
CutmixAlpha: 1.0
LabelSmoothing: 0.1
Scheduler: "None"

# --- Histopathology Specific Options ---
UseStainJitter: false
UseBackgroundRemoval: false
UseColorDeconvolution: false
```

### Key Configuration Parameters Explained

- `ModelName`, `Optimizer`, `LossFunction`: Accepting lists allows the script to automatically iterate through all
  combinations, creating separate output directories for each experiment.
- `UseAmp`: Enables Automatic Mixed Precision, halving memory usage and accelerating training.
- `UseEma`: Enables Exponential Moving Average of model weights, improving generalization.
- `UseMixup`: Blends images and labels to prevent overfitting.
- `UseColorDeconvolution`: Separates Hematoxylin and Eosin stains, highly beneficial for medical imaging.
- `UseBackgroundRemoval`: Masks out white background using Otsu's thresholding.

## Execution Instructions

### Step 1: Run the Script

Open your terminal, navigate to the script directory, and execute:

```bash
# Execute the main pipeline script.
python PyTorchPretrainedViTPipeline.py
```

*(Ensure your configuration file is named `config.pytorch.yaml` or specify it via `--config YourConfig.yaml`)*

### Step 2: Monitor Training

The script will print real-time training and validation metrics, automatically save the best model, and generate
visualizations upon completion.

## Experimental Results

To demonstrate the robustness and generalization capabilities of the pipeline, the model was evaluated on
the [Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data)
available on Kaggle. This dataset comprises 6 distinct classes: cardboard, glass, metal, paper, plastic, and trash.

### Evaluation Metrics

The model was tested on a held-out test set of 257 images. The pipeline achieved an overall **Accuracy of 93.39%** and a
**Macro F1-Score of 0.92**, indicating strong performance across all categories.

#### Classification Report

```text
              precision    recall  f1-score   support

   cardboard       1.00      0.88      0.94        41
       glass       0.94      0.98      0.96        51
       metal       0.95      1.00      0.98        41
       paper       0.92      0.95      0.93        60
     plastic       0.98      0.86      0.91        49
       trash       0.70      0.93      0.80        15

    accuracy                           0.93       257
   macro avg       0.92      0.93      0.92       257
weighted avg       0.94      0.93      0.93       257
```

#### Confusion Matrix

The following table illustrates the classification distribution across the 6 categories:

| Actual \ Predicted | cardboard | glass | metal | paper | plastic | trash |
|--------------------|-----------|-------|-------|-------|---------|-------|
| **cardboard**      | 36        | 0     | 0     | 5     | 0       | 0     |
| **glass**          | 0         | 50    | 1     | 0     | 0       | 0     |
| **metal**          | 0         | 0     | 41    | 0     | 0       | 0     |
| **paper**          | 0         | 0     | 0     | 57    | 1       | 2     |
| **plastic**        | 0         | 2     | 1     | 0     | 42      | 4     |
| **trash**          | 0         | 1     | 0     | 0     | 0       | 14    |

### Analytical Observations

- **Metal** and **Glass** exhibited near-perfect recall and precision, demonstrating robust feature extraction for
  rigid, reflective materials.
- **Cardboard** and **Plastic** showed high precision but slightly lower recall, suggesting that a minority of instances
  were misclassified as paper or trash due to visual similarities.
- **Trash** achieved the lowest precision (0.70), which is expected given its high visual variance and inherent overlap
  with other categories, though it maintained a high recall (0.93).

## Advanced Features & Architectures

### Supported Model Architectures

- `StandardViT`: The foundational Vision Transformer.
- `T2TViT`: Tokens-to-Token ViT for progressive tokenization.
- `HierarchicalViT`: Custom hierarchical transformer with patch merging.
- `CLIPViT`: Utilizes OpenAI's frozen CLIP visual encoder for zero-shot feature extraction.
- `SwinTransformer` / `SwinTransformerV2`: Hierarchical models using shifted windows.
- `DeiT`: Data-efficient Image Transformer for smaller datasets.
- `ConvNeXt` / `ConvNeXtV2`: Modernized hybrid CNN-Transformer architectures.
- `MaxViT`: Multi-axis Vision Transformer combining global and local attention.
- `BEiT`: Bidirectional Encoder representation from Image Transformers.
- `FastViT`: Ultra-fast mobile ViT combining CNN spatial mixing with global attention.
- `EVA02`: State-of-the-art model utilizing masked image modeling and CLIP distillation.

### Advanced Optimizers

- **Standard:** Adam, AdamW, SGD, RMSprop, RAdam.
- **Next-Generation:**
    - `Lion`: EvoLved Sign Momentum (saves memory, faster convergence).
    - `Prodigy`: Auto-tunes learning rate dynamically.
    - `ScheduleFreeAdamW`: Schedule-free optimization.
    - `Sophia`: Second-order optimizer using curvature information.

### Comprehensive Evaluation Metrics

The pipeline automatically calculates and exports:

- Confusion Matrices (Visual and CSV)
- ROC Curves and AUC-ROC
- Classification Reports (Precision, Recall, F1-Score)
- Macro, Micro, and Weighted averages for: Precision, Recall, F1, Accuracy, Specificity, Balanced Accuracy (BAC),
  Matthews Correlation Coefficient (MCC), Youden's Index, and Yule's Q.

## Output Structure & Interpretation

Upon completion, results are organized hierarchically:

```text
Results/
 ├── Exp-Swin-Prodigy-64-Focal/
 │   ├── Train/
 │   │   ├── TrainClassificationReport.txt
 │   │   ├── TrainCM.csv
 │   │   ├── TrainCM.png
 │   │   ├── TrainDetailedPredictions.csv
 │   │   └── TrainEvaluationMetrics.json
 │   ├── Val/
 │   │   ├── ValClassificationReport.txt
 │   │   ├── ValCM.png
 │   │   └── ...
 │   ├── Test/
 │   │   ├── TestClassificationReport.txt
 │   │   ├── TestCM.png
 │   │   └── ...
 │   ├── BestModel.pt              # The best model weights
 │   ├── ClassHistograms.png       # Class distribution across splits
 │   ├── ConfigUsed.yaml           # Exact configuration used
 │   └── TrainingHistory.png       # Loss and accuracy curves
```

### Key Output Files

- `BestModel.pt`: The trained model state dictionary, ready for deployment.
- `DetailedPredictions.csv`: Contains per-image predictions, actual labels, and prediction probabilities for granular
  error analysis.
- `EvaluationMetrics.json`: A structured JSON containing all calculated mathematical metrics.
- `TrainingHistory.png`: Visual plots of training and validation loss/accuracy over epochs.

## Example Console Output

```text
Starting ViT Image Classification Pipeline
Using device: cuda
Data Directory: ...
Output Directory: ...
Preparing data loaders...
Found 6 classes: ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
Train samples: 2054
Val samples: 257
Test samples: 257
Data loaders ready.
Class histograms saved to ./Results/ClassHistograms.png
============================================================
Starting Experiment: Exp-Swin-Prodigy-64-Focal
Output Directory: ./Results/Exp-Swin-Prodigy-64-Focal
============================================================
Epoch 1/500 - Train Loss: 0.9114 - Train Acc: 0.5971 - Val Loss: 0.8052 - Val Acc: 0.6119
  Saved new best model with validation loss = 0.8052
Epoch 2/500 - Train Loss: 0.7142 - Train Acc: 0.6850 - Val Loss: 0.5765 - Val Acc: 0.7761
  Saved new best model with validation loss = 0.5765
...
Epoch 12/500 - Train Loss: 0.8921 - Train Acc: 0.7812 - Val Loss: 0.9102 - Val Acc: 0.7150
  Early stopping triggered after 12 epochs. Best Val Loss: 0.3027
Training history plots saved to ./Results/TrainingHistory.png
Starting evaluation: savePlots=True | outputDir=./Results/Exp-Swin-Prodigy-64-Focal | prefix=Test
Confusion matrix saved to ./Results/Exp-Swin-Prodigy-64-Focal/Test/TestCM.png
Pipeline Complete | Results saved to ./Results/Exp-Swin-Prodigy-64-Focal
============================================================
All experiments completed.
```

## Explainability & XAI Features

To ensure transparency and interpretability in medical and high-stakes image classification, the pipeline includes a
dedicated Explainable AI (XAI) module. This module leverages Class Activation Mapping (CAM) techniques from
the [HMB Helpers Package](https://github.com/HossamBalaha/HMB-Helpers-Package) to visualize the regions of an image that
most significantly influence the model's predictions.

### Supported CAM Techniques

The integrated `CAMExplainerPyTorch` helper supports a comprehensive suite of attribution methods:

- **Gradient-based:** Grad-CAM, Grad-CAM++, XGrad-CAM, Layer-CAM, SmoothGrad-CAM++, Grad x Input.
- **Perturbation-based:** Occlusion, Ablation-CAM.
- **Activation-based:** Score-CAM, Eigen-CAM.
- **Other:** Saliency, SmoothGrad, Integrated Gradients.

### XAI Execution Scripts

To facilitate model interpretability, the repository includes two dedicated standalone scripts for generating
Explainable AI visualizations.

#### 1. `PyTorchPretrainedViTXAI.py` (Batch Processing by Method)

**Description:**  
This script automates the generation of Class Activation Mapping (CAM) visualizations for a trained model across a
specified dataset split. It iterates through multiple attribution methods and saves the resulting heatmaps organized by
technique (e.g., separate outputs for Grad-CAM, SmoothGrad, etc.) using the `hmb-helpers` package.

**When to Use:**  
Use this script when you need to process an entire dataset split efficiently and want the XAI outputs organized by
technique. It is ideal for large-scale dataset analysis or when you need to extract raw masks and individual
visualizations separated by method.

**How to Use:**

1. Ensure you have a trained model checkpoint (e.g., `BestModel.pt`) and a properly organized dataset.
2. Open `PyTorchPretrainedViTXAI.py` and update the following variables in the `if __name__ == "__main__":` block:
    - `datasetPath`: The absolute path to your dataset directory.
    - `modelCheckpointPath`: The absolute path to your trained `.pt` model weights.
    - `outputDir`: The desired directory for saving the XAI visualizations.
    - `modelName` and `numClasses`: The architecture name and number of classes for your specific model.
    - `classNamesMapping`: A dictionary mapping integer class indices to their respective CamelCase string names (e.g.,
      `{0: "Healthy", 1: "Tumor"}`).
3. Execute the script from your terminal:
   ```bash
   python PyTorchPretrainedViTXAI.py
   ```
4. The script will automatically load the model, sample up to 25 images per class from the specified split, and generate
   comprehensive CAM overlays for all supported methods.

#### 2. `AdvancedXAIOnDataset.py` (Side-by-Side Visual Comparison)

**Description:**  
This script generates a **single, combined side-by-side figure** for each image, displaying the original image alongside
the XAI masks from all selected methods in a single grid row.

**When to Use:**  
Use this script when you need to visually compare how different XAI methods highlight the exact same image. It is highly
recommended for generating publication-ready figures, presentation slides, or detailed case studies where direct
method-to-method comparison is required.

**How to Use:**

1. Ensure you have a trained model checkpoint and a properly organized dataset.
2. Open `AdvancedXAIOnDataset.py` and update the following variables in the `if __name__ == "__main__":` block:
    - `datasetPath`: The absolute path to your dataset directory.
    - `modelCheckpointPath`: The absolute path to your trained `.pt` model weights.
    - `outputDir`: The desired directory for saving the combined XAI figures.
    - `modelName`, `numClasses`, and `effectiveImageSize`: The architecture details.
    - `classNamesMapping`: A dictionary mapping integer class indices to their respective CamelCase string names.
    - `methods`: A list of specific XAI methods you want to include in the side-by-side comparison (e.g.,
      `["integratedgradients", "smoothgrad", "vitgradcam"]`).
3. Execute the script from your terminal:
   ```bash
   python AdvancedXAIOnDataset.py
   ```
4. The script will automatically load the model, sample images per class, and save high-resolution `.png` and `.pdf`
   figures containing the original image and all selected XAI overlays side-by-side.

### Key Differences Summary

| Feature           | `PyTorchPretrainedViTXAI.py`                            | `AdvancedXAIOnDataset.py`                                       |
|:------------------|:--------------------------------------------------------|:----------------------------------------------------------------|
| **Output Format** | Individual files or folders per XAI technique.          | Single combined grid figure (Original + all methods) per image. |
| **Best For**      | Large-scale batch processing and dataset-wide analysis. | Direct visual comparison, publications, and presentations.      |
| **Customization** | Processes all available methods in the helper package.  | Allows selecting a specific subset of methods for the grid.     |

## Appendix: Inference Example

Below is a formal example demonstrating how to load the trained model for standard inference, strictly adhering to the
repository's coding standards.

```python
# Import the torch module for deep learning operations.
import torch

# Import the torchvision transforms module for image preprocessing.
from torchvision import transforms

# Import the PIL Image module for image loading and manipulation.
from PIL import Image

# Import the standard vision transformer model from the pipeline module.
from PyTorchPretrainedViTPipeline import StandardViT


# Define the inference function for making predictions on a single image.
def RunInference(
  modelPath: str,
  imagePath: str,
  numClasses: int,
) -> None:
  # Load the saved model weights from the specified file path.
  modelStateDict = torch.load(modelPath, map_location="cpu")

  # Initialize the standard vision transformer architecture with the correct number of classes.
  model = StandardViT(numClasses=numClasses)

  # Load the state dictionary into the initialized model architecture.
  model.load_state_dict(modelStateDict)

  # Set the model to evaluation mode to disable dropout and batch normalization updates.
  model.eval()

  # Check if the model is successfully initialized and ready for inference.
  if (model is not None):
    # Print a success message to the console indicating the model is loaded.
    print("Model loaded successfully.")

  # Define the image preprocessing transform pipeline.
  transformPipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
  ])

  # Load the input image from the specified file path.
  inputImage = Image.open(imagePath).convert("RGB")

  # Apply the preprocessing transform pipeline to the input image.
  processedImage = transformPipeline(inputImage)

  # Add a batch dimension to the processed image tensor.
  batchedImage = processedImage.unsqueeze(0)

  # Perform the forward pass to obtain the model predictions.
  with torch.no_grad():
    # Get the raw output logits from the model.
    outputLogits = model(batchedImage)

  # Apply the softmax function to convert logits to probabilities.
  probabilities = torch.softmax(outputLogits, dim=1)

  # Get the predicted class index with the highest probability.
  predictedClass = torch.argmax(probabilities, dim=1).item()

  # Print the final predicted class to the console.
  print(f"Predicted class index: {predictedClass}")
```

---

## License

This project is distributed under a custom **Academic and Non-Commercial License**.

The software is provided free of charge for academic, research, and non-commercial purposes. Any commercial use,
exploitation, or distribution of the Software is strictly prohibited without prior written permission from the copyright
holder.

For the full legal terms and conditions, please refer to the [`LICENSE`](LICENSE) file located in the root of this
repository.

For commercial licensing inquiries, please contact the author.

---

## 📬 Contact

This repository is prepared by `Hossam Magdy Balaha`. For any questions or inquiries, please contact me using the
contact information available on my CV at the following
link: [https://hossambalaha.github.io/](https://hossambalaha.github.io/)

---
*Last Updated: August 2026*
