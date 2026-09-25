# Comprehensive Guide to Vision Transformer (ViT) Image Classification Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?logo=nvidia&logoColor=white" alt="CUDA Accelerated">
  <img src="https://img.shields.io/badge/HMB%20Helpers-Integrated-2D963D?logo=python&logoColor=white" alt="HMB Helpers">
  <img src="https://img.shields.io/badge/License-Academic%20and%20Non--Commercial-orange" alt="License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architectures-EVA02%20%7C%20Swin%20%7C%20DeiT%20%7C%20ConvNeXt-8A2BE2" alt="Supported Architectures">
  <img src="https://img.shields.io/badge/XAI-GradCAM%20%7C%20RISE%20%7C%20Integrated%20Gradients-FFD700" alt="Explainable AI">
</p>

<p align="center">
  <a href="https://hossambalaha.github.io/">
    <img src="https://img.shields.io/badge/Author-Hossam%20Magdy%20Balaha-1DA1F2?logo=github" alt="Author">
  </a>
</p>

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
- [Advanced Model Evaluation & Analysis](#advanced-model-evaluation--analysis)
- [Output Structure & Interpretation](#output-structure--interpretation)
- [Example Console Output](#example-console-output)
- [Explainability & XAI Features](#explainability--xai-features)
- [Appendix: Inference Example](#appendix-inference-example)

---

## Introduction

This repository provides a robust, production-grade PyTorch pipeline for image classification utilizing Vision
Transformers (ViT), hybrid classical-quantum networks, and other state-of-the-art architectures, including
Kolmogorov-Arnold Networks (KANs), Neural Ordinary Differential Equations (Neural ODEs), Spiking Neural Networks (SNNs),
Hypernetworks, Liquid State Space Vision Transformers (LiquidSSM-ViT), Test-Time Evolving Transformers (TTT-ViT), Tensor
Network Entangled Vision Transformers (TNE-ViT), and Diffusion-Prior Energy Vision Transformers (DiffEnergy-ViT).
It is specifically designed to accommodate students and researchers, offering an accessible yet
highly configurable framework. The pipeline supports automatic dataset splitting, advanced data preprocessing (including
histopathology-specific augmentations), multiple cutting-edge models, modern optimizers, comprehensive evaluation
metrics, result aggregation utilities, integrated Explainable AI (XAI) capabilities, and native Quantum Transfer
Learning via PennyLane. The pipeline also includes experimental support for Topological Wasserstein Loss to enforce
meaningful, linearly separable class clustering in the latent space. (Note: Full integration requires custom criterion
wrapping, as the standard HMB Training Pipeline currently falls back to standard classification loss for this feature.)

## Author Information

- **Author:** Hossam Magdy Balaha
- **Online CV:** [https://hossambalaha.github.io/](https://hossambalaha.github.io/)

## Conceptual Foundations

Prior to examining the source code, it is advantageous to comprehend the underlying computational processes. The
following curated video resources, articles, and tutorials are recommended to establish a foundational understanding of
both the core mechanics and the advanced techniques utilized in this pipeline.

### General Deep Learning & PyTorch Fundamentals

1. [But what is a neural network? | Deep learning chapter 1 (3Blue1Brown)](https://www.youtube.com/watch?v=aircAruvnKk)
2. [A Gentle Introduction to Machine Learning](https://www.youtube.com/watch?v=Gv9_4yMHFhI)
3. [PyTorch Crash Course: Deep Learning in Python](https://www.youtube.com/watch?v=uq7sbUlIDR8)
4. [PyTorch in 1 Hour](https://www.youtube.com/watch?v=r1bquDz5GGA)

### Vision Transformers (ViT) & Modern Architectures

1. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://www.youtube.com/watch?v=TrdevFK_am4)
2. [Vision Transformer Quick Guide - Theory and Code](https://www.youtube.com/watch?v=j3VNqtJUoz0)
3. [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://www.youtube.com/watch?v=tFYxJZBAbE8)
4. [Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet](https://www.youtube.com/watch?v=eaZt9asVYH0)

### Convolutional Neural Network (CNN) Foundations

1. [PyTorch CNN Part 1](https://www.youtube.com/watch?v=g6eHItPHd7k)
2. [PyTorch CNN Part 2](https://www.youtube.com/watch?v=_QpiHNykR9k)

### Explainable AI (XAI) & Interpretability *(Crucial for Steps 3A & 3B)*

1. [ACM AI Reading Group 2/20 Session: Sanity Checks for Saliency Maps](https://www.youtube.com/watch?v=naxbilQxxPM)
2. [Grad-CAM](https://www.youtube.com/watch?v=COjUB9Izk6E)
3. [Integrated Gradients Explained — Theory, Axioms & Python Implementation](https://www.youtube.com/watch?v=CpiX4WurL7w)
4. [RISE (randomized input sampling for explanation of black box models)](https://www.youtube.com/watch?v=VjchURwP1j8)

### Model Calibration & Uncertainty Quantification *(Crucial for Step 5)*

1. [PR-075: On Calibration of Modern Neural Networks (2017)](https://www.youtube.com/watch?v=odNHEkfJAc4)
2. [Model Calibration - Estimated Calibration Error (ECE) Explained](https://www.youtube.com/watch?v=NDY2fH1FitQ)

### Robustness, Perturbations, and Domain Shift *(Crucial for Step 6)*

1. [PR-277: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://www.youtube.com/watch?v=EE4BxrAbNM8)
2. [ImageNet-D: Benchmarking Neural Network Robustness on Diffusion Synthetic Object @CVPR2024 Highlight](https://www.youtube.com/watch?v=CQm2oDfCvR8)

### Representation Learning & Latent Space Analysis *(Crucial for Step 8)*

1. [How to Use t-SNE Effectively: Distill's Classic, Read and Highlighted](https://www.youtube.com/watch?v=4cDoRtr5aLo)
2. [UMAP Dimension Reduction, Main Ideas!!!](https://www.youtube.com/watch?v=eN0wFzBA4Sc)
3. [Cluster Analysis in Python - Silhouette, Calinski Harabasz, and Davies Bouldin for KMeans](https://www.youtube.com/watch?v=MHnoWsBJpeM)

### Statistical Significance in Machine Learning *(Crucial for Step 7)*

1. [Statistical Significance and p-Values Explained Intuitively](https://www.youtube.com/watch?v=DAkJhY2zQ3c)
2. [What is a Raincloud Plot? [Simply explained]](https://www.youtube.com/watch?v=ituWaiJu3nI)

### Digital Pathology & Histopathology Preprocessing *(Crucial for Medical Imaging)*

1. [122 - Normalizing H&E images and digitally separating Hematoxylin and Eosin components](https://www.youtube.com/watch?v=yUrwEYgZUsA)
2. [304 - Augmentation of histology images to train stain-agnostic deep learning models](https://www.youtube.com/watch?v=SuDtHqtC5OE)

### Topological Data Analysis & Advanced Loss Functions *(Crucial for `WassersteinTopologicalLoss`)*

1. [Persistent Homology | Introduction & Python Example Code](https://www.youtube.com/watch?v=5ezFcy9CIWE)
2. [Wasserstein Distance & Optimal Transport — Fully Explained](https://www.youtube.com/watch?v=88ONbF_b3VE)
3. [Contrastive Learning - 5 Minutes with Cyrill](https://www.youtube.com/watch?v=sftIkJ8MYL4)

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

# Install PennyLane for quantum machine learning and hybrid classical-quantum models.
pip install pennylane

# Install the HMB Helpers Package for Explainable AI (XAI), efficiency profiling, and advanced utility functions.
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
  - "QuantumResNet"

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

# --- Pipeline Control Parameters ---
# Criterion to judge the best model for checkpointing and early stopping ("val_loss", "val_accuracy", "both").
JudgeBy: "val_accuracy"

# Save a checkpoint every N epochs. Set to null to disable periodic saving.
SaveEvery: null

# Maximum gradient norm for clipping. Set to null to disable gradient clipping.
MaxGradNorm: null

# Whether to resume training from the last checkpoint if it exists.
ResumeFromCheckpoint: false

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

# --- Topological Wasserstein Loss Hyperparameters ---
UseTopoWassersteinLoss: true
Epsilon: 0.1
LambdaTopo: 0.1
LambdaContrastive: 0.1

# --- Multi-Trial Execution ---
Seeds: [ 42, 43, 44, 45, 46 ]
```

### Key Configuration Parameters Explained

- `ModelName`, `Optimizer`, `LossFunction`: Accepting lists allows the script to automatically iterate through all
  combinations, creating separate output directories for each experiment.
- `UseAmp`: Enables Automatic Mixed Precision, halving memory usage and accelerating training.
- `UseEma`: Enables Exponential Moving Average of model weights, improving generalization.
- `UseMixup`: Blends images and labels to prevent overfitting.
- `UseColorDeconvolution`: Separates Hematoxylin and Eosin stains, highly beneficial for medical imaging.
- `UseBackgroundRemoval`: Masks out white background using Otsu's thresholding.
- `UseTopoWassersteinLoss`: Enables the novel Topological Wasserstein Loss for improved feature space distributional
  alignment and spatial constraints.
- `Epsilon`, `LambdaTopo`, `LambdaContrastive`: Hyperparameters controlling the entropic regularization and weighting
  factors for the topological spatial penalty and contrastive loss.
- `Seeds`: A list of random seeds to ensure statistical significance through multi-trial execution.
- `UsePretrainedCustomModels`: Enables partial ImageNet weight transfer for custom from-scratch architectures (e.g.,
  KAN, Neural ODE, Spiking).
- `PretrainedModelName`: Specifies the `timm` model name (e.g., `vit_base_patch16_224`) to use as the source for
  pretrained weights when `UsePretrainedCustomModels` is enabled.
- `JudgeBy`: Criterion used to determine the "best" model for checkpointing and early stopping (e.g., `"val_loss"`,
  `"val_accuracy"`, or `"both"`).
- `SaveEvery`: Saves a model checkpoint every N epochs. Set to `null` to disable periodic saving and rely only on the
  best-model checkpoint.
- `MaxGradNorm`: Maximum gradient norm for gradient clipping to prevent exploding gradients. Set to `null` to disable.
- `ResumeFromCheckpoint`: If `true`, automatically resumes training from the latest saved checkpoint in the output
  directory.

## Execution Instructions

### Step 1: Run the Main Pipeline

Open your terminal, navigate to the script directory, and execute:

```bash
# Execute the main pipeline script.
python Step1PyTorchPretrainedViTPipeline.py
```

*(Ensure your configuration file is named `config.pytorch.yaml` or specify it via `--config YourConfig.yaml`)*

### Step 2: Monitor Training

The script will print real-time training and validation metrics, automatically save the best model, and generate
visualizations upon completion.

### Step 3: Result Aggregation (For Multi-Seed Experiments)

After running multiple seeds, aggregate the results to compute statistical summaries required for rigorous reporting:

```bash
# 1. Collect metrics from all experiment folders into consolidated CSVs.
python Step2ACollectResults.py

# 2. Aggregate multi-trial results to compute mean ± standard deviation across seeds.
python Step2BAggregateMultiTrialResults.py
```

### Step 4: Execute Advanced Model Analysis (Optional but Recommended)

After the main pipeline completes, you can run the dedicated analysis scripts to generate supplementary materials
useful for advanced research and detailed reporting:

```bash
# 1. Inference Efficiency Profiling
python Step4InferenceEfficiencyProfiling.py

# 2. Model Calibration Analysis
python Step5ModelCalibrationAnalysis.py

# 3. Robustness & Perturbation Analysis
python Step6RobustnessPerturbationAnalysis.py

# 4. Cross-Experiment Statistical Analysis (Requires multiple seed folders)
python Step7CrossExperimentStatisticalAnalysis.py

# 5. Representation Learning & Latent Space Analysis
python Step8RepresentationLearningEmbeddings.py

# 6. Minority Class Precision-Recall Curves
python Step9MinorityClassPrecisionRecallCurves.py
```

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
- `EVA02Large`: High-resolution (448x448) variant of EVA-02 for fine-grained feature extraction.
- `ConvNeXtLargeCLIP`: ConvNeXt Large model with robust CLIP (LAION-2B) pre-training.
- `SwinTransformerLarge384`: Swin Large model optimized for 384x384 high-resolution input.
- `EfficientNetV2Large`: Highly efficient, robust feature extraction model.
- `QuantumResNet`: A hybrid classical-quantum architecture utilizing a pre-trained ResNet152 feature extractor, coupled
  with a custom dimensionality reduction block and a PennyLane variational quantum circuit for advanced latent space
  classification.
- `VisionKANModel`: Kolmogorov-Arnold Network-based Vision Transformer replacing standard MLPs with learnable KAN layers
  for superior interpretability and parameter efficiency.
- `NeuralODEViTModel`: Continuous-depth Vision Transformer modeling hidden state dynamics via Neural Ordinary
  Differential Equations for adaptive computation and memory efficiency.
- `SpikingViTModel`: Spiking Vision Transformer incorporating biologically plausible Leaky Integrate-and-Fire neurons
  for ultra-low power consumption and event-driven processing.
- `HypernetworkViTModel`: Hypernetwork-based Vision Transformer utilizing a secondary network to dynamically generate
  primary model weights for extreme compression and rapid adaptation.
- `LiquidSSMViTModel`: Liquid State Space Vision Transformer merging Liquid Time-Constant ODEs with linear-time State
  Space Models for continuous-time, adaptive receptive fields.
- `TestTimeEvolvingViTModel`: Test-Time Evolving Transformer incorporating a self-supervised auxiliary head for
  continuous feature adaptation during inference.
- `TensorNetworkEntangledViTModel`: Tensor Network Entangled Vision Transformer replacing standard attention with Matrix
  Product States for exponential parameter compression.
- `DiffusionPriorEnergyViTModel`: Diffusion-Prior Energy Vision Transformer utilizing Energy-Based Models for
  unparalleled out-of-distribution detection.
- `FractalResonanceViTModel`: Fractal-Resonance Vision Transformer utilizing harmonic resonance and fractal
  dimensionality for multi-scale feature extraction without standard convolutions.
- `TopologicalQuantumViTModel`: Topological Quantum Vision Transformer computing attention weights based on persistent
  homology and Betti numbers for extreme robustness to spatial deformations.
- `HolographicInterferenceViTModel`: Holographic Interference Vision Transformer encoding spatial features as phase and
  amplitude holograms for ultra-dense parallel processing.
- `NeuromorphicLiquidStateViTModel`: Neuromorphic Liquid State Machine combining reservoir computing with spiking
  dynamics for extreme one-shot adaptation and temporal-spatial continuity.

**Pretrained Weight Transfer for Custom Models:** All custom from-scratch architectures
(VisionKAN, NeuralODE, Spiking, Hypernetwork, LiquidSSM, TestTimeEvolving,
TensorNetworkEntangled, DiffusionPriorEnergy, FractalResonance, TopologicalQuantum,
HolographicInterference, NeuromorphicLiquidState) support partial ImageNet pretrained
weight transfer. When enabled, the patch embedding, class token, positional encoding,
layer normalization, and classification head are initialized from a pretrained ViT-Base
model, while the custom architectural blocks remain randomly initialized. This provides
a significant performance boost on small-to-medium datasets by utilizing rich visual
priors learned from 1.2 million ImageNet images.

### Advanced Optimizers

- **Standard:** Adam, AdamW, SGD, RMSprop, RAdam.
- **Next-Generation:**
    - `Lion`: EvoLved Sign Momentum (saves memory, faster convergence).
    - `Prodigy`: Auto-tunes learning rate dynamically.
    - `ScheduleFreeAdamW`: Schedule-free optimization.
    - `Sophia`: Second-order optimizer using curvature information (implemented natively to avoid package conflicts).

### Comprehensive Evaluation Metrics

The pipeline automatically calculates and exports:

- Confusion Matrices (Visual and CSV)
- ROC Curves and AUC-ROC
- Classification Reports (Precision, Recall, F1-Score)
- Macro, Micro, and Weighted averages for: Precision, Recall, F1, Accuracy, Specificity, Balanced Accuracy (BAC),
  Matthews Correlation Coefficient (MCC), Youden's Index, and Yule's Q.

## Advanced Model Evaluation & Analysis

The pipeline includes dedicated scripts for comprehensive model evaluation beyond standard accuracy. These steps ensure
statistical rigor, robustness, and deep interpretability for research and production environments.

### 1. Inference Efficiency Profiling (`Step4InferenceEfficiencyProfiling.py`)

- **Purpose:** Quantifies the computational cost of models to establish Pareto-optimal trade-offs between accuracy and
  efficiency, a mandatory metric for deployment feasibility.
- **Output:** Multi-metric Pareto fronts, parameter counts, latency comparisons, stacked memory breakdowns, GFLOPs,
  throughput, and efficiency summary dashboards.
- **Usage:** Profiles specified models using a dummy input by utilizing the modular `ProfileModelEfficiency` function
  from `HMB.Examples.ModelsInferenceEfficiencyProfiling`. It then utilizes `HMB.PyTorchModelMemoryProfiler` and
  `HMB.PlotsHelper.EfficiencyPlotter` to generate high-quality efficiency visualizations and detailed CamelCase
  JSON profiles (e.g., `ModelNameDetailedProfile.json`).

### 2. Model Calibration Analysis (`Step5ModelCalibrationAnalysis.py`)

- **Purpose:** Evaluates whether the model's confidence aligns with its actual accuracy, a critical requirement for safe
  clinical deployment.
- **Output:** Expected Calibration Error (ECE) metric (printed to the console) and a high-quality Reliability
  Diagram (`ReliabilityDiagram.pdf`).
- **Usage:** Automatically evaluates the test set using `HMB.PerformanceMetrics.ComputeECEPlotReliability` to detect
  overconfidence or underconfidence in predictions by binning probabilities and visualizing the calibration curve.

### 3. Robustness & Perturbation Analysis (`Step6RobustnessPerturbationAnalysis.py`)

- **Purpose:** Quantifies model resilience against real-world imaging degradations to ensure safe and reliable clinical
  deployment.
- **Output:** Mean Corruption Error (mCE) metrics, high-resolution accuracy heatmaps across severity levels, and
  Expected Calibration Error (ECE) heatmaps under perturbation (saved as both `.png` and `.pdf` for detailed reporting).
- **Usage:** Executes controlled corruptions (Gaussian noise, brightness shifts, JPEG compression, speckle noise, and
  contrast variations) across 5 severity levels on a test subset. It utilizes a robust prediction callable that
  seamlessly handles diverse input formats (Tensor, NumPy, PIL) via
  `HMB.PyTorchHelper.EvaluateModelOnPerturbations`.

### 4. Cross-Experiment Statistical Analysis (`Step7CrossExperimentStatisticalAnalysis.py`)

- **Purpose:** Provides mathematical proof that performance differences across multiple random seeds are statistically
  significant, rather than due to random chance.
- **Output:** High-quality Raincloud, Box, and Violin plots (generated via
  `HMB.StatisticalAnalysisHelper.PlotMetrics`),
  alongside console-printed pairwise t-test results (comparing the baseline seed to subsequent seeds) with Cohen's *d*
  effect sizes.
- **Usage:** Aggregates a specified metric (e.g., `WeightedAccuracy`) from `TestEvaluationMetrics.json` files across all
  seed directories to validate reproducibility and statistical significance.

### 5. Representation Learning & Latent Space Analysis (`Step8RepresentationLearningEmbeddings.py`)

- **Purpose:** Proves that the model learns meaningful, linearly separable class clusters in the latent space, providing
  deep insights into feature representation and model decision boundaries.
- **Output:** `Embeddings.pkl` (raw feature vectors), `Embeddings_Metadata.csv` (true labels, predictions, and
  filenames), and comprehensive `tSNE_Analysis/` and `UMAP_Analysis/` directories. These directories contain interactive
  Plotly HTML files, static high-quality figures, cluster quality metrics (Silhouette, Davies-Bouldin,
  Calinski-Harabasz), misclassification highlighting, and centroid annotations.
- **Usage:** Extracts penultimate layer embeddings (automatically handling both ViT CLS tokens and CNN spatial
  averaging)   and utilizes `HMB.ExplainabilityHelper.TSNEFeaturesExplainability` and `UMAPFeaturesExplainability` to
  generate the advanced visualizations and statistical metrics.

### 6. Minority Class Precision-Recall Curves (`Step9MinorityClassPrecisionRecallCurves.py`)

- **Purpose:** Addresses class imbalance by evaluating the Area Under the Precision-Recall Curve (AUPRC) for
  underrepresented classes, which ROC-AUC can be overly optimistic about.
- **Output:** Highlighted Precision-Recall Curves (PRC) emphasizing minority classes (saved as both `.pdf` and `.png`),
  alongside console-printed overall Weighted and Macro F1 scores computed via
  `HMB.PerformanceMetrics.CalculatePerformanceMetrics`.
- **Usage:** Parses the `DetailedPredictions.csv` to compute and visualize per-class AUPRC, ensuring the model does not
  achieve high accuracy by simply ignoring rare classes.

## Output Structure & Interpretation

Upon completion, results are organized hierarchically. When advanced analysis scripts are executed, additional
directories are populated:

```text
Results/
 ├── Exp-EVA02-AdamW-16-CrossEntropy/
 │   ├── Seed-42/
 │   │   ├── Train/
 │   │   │   └── ... (Training metrics and plots)
 │   │   ├── Val/
 │   │   │   └── ... (Validation metrics and plots)
 │   │   ├── Test/
 │   │   │   ├── TestClassificationReport.txt
 │   │   │   ├── TestCM.png
 │   │   │   ├── TestDetailedPredictions.csv
 │   │   │   └── TestEvaluationMetrics.json
 │   │   ├── Calibration/
 │   │   │   └── ReliabilityDiagram.pdf
 │   │   ├── Robustness/
 │   │   │   ├── RobustnessReport.json
 │   │   │   ├── AccuracyHeatmap.png
 │   │   │   └── EceHeatmap.png
 │   │   ├── Embeddings/
 │   │   │   ├── Embeddings.pkl
 │   │   │   ├── Embeddings_Metadata.csv
 │   │   │   ├── tSNE_Analysis/          # t-SNE plots and cluster metrics
 │   │   │   └── UMAP_Analysis/          # UMAP plots and cluster metrics
 │   │   ├── PRC_Analysis/
 │   │   │   ├── PrecisionRecallCurves.pdf
 │   │   │   └── PrecisionRecallCurves.png
 │   │   ├── EfficiencyProfiling/        
 │   │   │   ├── EVA02DetailedProfile.json
 │   │   │   └── EfficiencyDashboard.png
 │   │   ├── BestModel.pt                # The best model weights
 │   │   ├── ClassHistograms.png         # Class distribution across splits
 │   │   ├── ConfigUsed.yaml             # Exact configuration used
 │   │   └── TrainingHistory.png         # Loss and accuracy curves
 │   └── StatisticalAnalysis/            # Aggregated across all seeds
 │       ├── WeightedAccuracy_Raincloud.pdf
 │       ├── WeightedAccuracy_Boxplot.pdf
 │       └── WeightedAccuracy_Violin.pdf
 ├── ValResults.csv                      
 ├── TestResults.csv                     
 ├── AggregatedResultsSummary.csv        
 └── RawResults.csv                      
```

### Key Output Files

- `BestModel.pt`: The trained model state dictionary, ready for deployment.
- `DetailedPredictions.csv`: Contains per-image predictions, actual labels, and prediction probabilities for granular
  error analysis.
- `EvaluationMetrics.json`: A structured JSON containing all calculated mathematical metrics.
- `TrainingHistory.png`: Visual plots of training and validation loss/accuracy over epochs.
- `AggregatedResultsSummary.csv`: Consolidated mean ± standard deviation metrics across all random seeds for detailed
  reporting and statistical analysis.

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
Starting Experiment: Exp-Swin-Prodigy-64-Focal | Seed: 42
Output Directory: ./Results/Exp-Swin-Prodigy-64-Focal/Seed-42
============================================================
Epoch 1/500 - Train Loss: 0.9114 - Train Acc: 0.5971 - Val Loss: 0.8052 - Val Acc: 0.6119
  Saved new best model with validation loss = 0.8052
Epoch 2/500 - Train Loss: 0.7142 - Train Acc: 0.6850 - Val Loss: 0.5765 - Val Acc: 0.7761
  Saved new best model with validation loss = 0.5765
...
Epoch 12/500 - Train Loss: 0.8921 - Train Acc: 0.7812 - Val Loss: 0.9102 - Val Acc: 0.7150
  Early stopping triggered after 12 epochs. Best Val Loss: 0.3027
Training history plots saved to ./Results/TrainingHistory.png
Starting evaluation: savePlots=True | outputDir=./Results/Exp-Swin-Prodigy-64-Focal/Seed-42 | prefix=Test
Confusion matrix saved to ./Results/Exp-Swin-Prodigy-64-Focal/Seed-42/Test/TestCM.png
Pipeline Complete | Results saved to ./Results/Exp-Swin-Prodigy-64-Focal/Seed-42
============================================================
All experiments completed.
```

## Explainability & XAI Features

To ensure transparency and interpretability in medical and high-stakes image classification, the pipeline includes a
dedicated Explainable AI (XAI) module. This module utilizes Class Activation Mapping (CAM) techniques from
the [HMB Helpers Package](https://github.com/HossamBalaha/HMB-Helpers-Package) to visualize the regions of an image that
most significantly influence the model's predictions.

### Supported CAM Techniques

The integrated `CAMExplainerPyTorch` helper supports a comprehensive suite of attribution methods:

- **Gradient-based:** Grad-CAM, Grad-CAM++, XGrad-CAM, Layer-CAM, SmoothGrad-CAM++, Grad x Input.
- **Perturbation-based:** Occlusion, Ablation-CAM.
- **Activation-based:** Score-CAM, Eigen-CAM.
- **Other:** Saliency, SmoothGrad, Integrated Gradients, RISE, Feature Ablation, ViT Grad-CAM, ViT XGrad-CAM, ViT
  Eigen-CAM.

### XAI Execution Scripts

To facilitate model interpretability, the repository includes two dedicated standalone scripts for generating
Explainable AI visualizations.

#### 1. `Step3APyTorchPretrainedViTXAI.py` (Batch Processing by Method)

**Description:**  
This script automates the generation of Class Activation Mapping (CAM) visualizations for a trained model across a
specified dataset split. It iterates through multiple attribution methods and saves the resulting heatmaps organized by
technique (e.g., separate outputs for Grad-CAM, SmoothGrad, etc.) using the `hmb-helpers` package.

**When to Use:**  
Use this script when you need to process an entire dataset split efficiently and want the XAI outputs organized by
technique. It is ideal for large-scale dataset analysis or when you need to extract raw masks and individual
visualizations separated by method.

**How to Use:**

1. Ensure you have a trained model checkpoint and a properly organized dataset.
2. Open `Step3APyTorchPretrainedViTXAI.py` and update the following variables in the `if __name__ == "__main__":` block:
    - `datasetPath`: The absolute path to your dataset directory.
    - `modelCheckpointPath`: The absolute path to your trained `.pt` model weights.
    - `outputDir`: The desired directory for saving the XAI visualizations.
    - `modelName` and `numClasses`: The architecture name and number of classes for your specific model.
    - `classNamesMapping`: A dictionary mapping integer class indices to their respective CamelCase string names (e.g.,
      `{0: "Healthy", 1: "Tumor"}`).
3. Execute the script from your terminal:
   ```bash
   python Step3APyTorchPretrainedViTXAI.py
   ```
4. The script will automatically load the model, sample up to 25 images per class from the specified split, and generate
   comprehensive CAM overlays for all supported methods.

#### 2. `Step3BAdvancedXAIOnDataset.py` (Side-by-Side Visual Comparison)

**Description:**  
This script generates a **single, combined side-by-side figure** for each image, displaying the original image alongside
the XAI masks from all selected methods in a single grid row.

**When to Use:**  
Use this script when you need to visually compare how different XAI methods highlight the exact same image.
It is highly recommended for generating high-quality figures, presentation slides, or detailed case studies where direct
method-to-method comparison is required.

**How to Use:**

1. Ensure you have a trained model checkpoint and a properly organized dataset.
2. Open `Step3BAdvancedXAIOnDataset.py` and update the following variables in the `if __name__ == "__main__":` block:
    - `datasetPath`, `modelCheckpointPath`, `outputDir`, `modelName`, `numClasses`, and `effectiveImageSize`.
    - `classNamesMapping`: A dictionary mapping integer class indices to their respective CamelCase string names.
    - `methods`: A list of specific XAI methods you want to include in the side-by-side comparison (e.g.,
      `["integratedgradients", "smoothgrad", "vitgradcam"]`).
3. Execute the script from your terminal:
   ```bash
   python Step3BAdvancedXAIOnDataset.py
   ```
4. The script will automatically load the model, sample images per class, and save high-resolution `.png` and `.pdf`
   figures containing the original image and all selected XAI overlays side-by-side.

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
from Step1PyTorchPretrainedViTPipeline import StandardViT


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
*Last Updated: September 2026*
