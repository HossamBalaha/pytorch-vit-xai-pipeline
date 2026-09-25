import os  # Import the operating system module.
import cv2  # Import the OpenCV module for image processing.
import copy  # Import the copy module for deep copying.
import csv  # Import the csv module for writing CSV files.
import json  # Import the json module for data serialization.
import yaml  # Import the yaml module for configuration parsing.
import clip  # Import the clip library for CLIP models.
import timm  # Import the timm library for Swin Transformer.
import numpy  # Import the numpy module for numerical operations.
import seaborn  # Import the seaborn module for statistical data visualization.
import time  # Import the time module for timing operations.
import torch  # Import the torch module for deep learning.
import tifffile  # Import the tifffile module for reading TIFF images.
import torch.nn  # Import the neural network module from torch.
import torch.optim  # Import the optim module from torch.
from tqdm import tqdm  # Import tqdm for progress bars.
from pathlib import Path  # Import the Path class from pathlib.
import matplotlib.pyplot as plt  # Import the pyplot module from matplotlib.
import torch.nn.functional as F  # Import the functional module from torch.
from sklearn.metrics import *  # Import metric functions from sklearn.
from typing import Dict, List, Tuple, Optional, Union, Any  # Import typing utilities.
from torchvision import datasets  # Import datasets from torchvision.
from torchvision import transforms  # Import transforms from torchvision.
from PIL import Image  # Import the Image module from PIL.
# Import the Dataset, DataLoader, and WeightedRandomSampler utilities from torch.
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from timm.loss import LabelSmoothingCrossEntropy
import torchvision  # Import the torchvision module for models and datasets.
from HMB.Utils import fprint, NumpyEncoder
from HMB.Initializations import SeedEverything
from HMB.PyTorchClassificationModelsZoo import *
from HMB.PyTorchHelper import (
  SophiaG,
  GetOptimizer,
  CreateTimmModel,
  EnableMixedPrecision,
  EarlyStopping,
  CheckpointSaver,
  LoadModel,
  SavePyTorchDict,
  LoadPyTorchDict,
  MixupFn,
  MixupCriterion,
  ExponentialMovingAverage
)
from HMB.WSIHelper import ColorDeconvolution, StainJitter
from HMB.PyTorchClassificationLosses import (
  WassersteinTopologicalLoss,
  CrossEntropyLossWrapper,
  LabelSmoothingCrossEntropy as LSCE_Loss,
  FocalLossRobust
)
from HMB.PerformanceMetrics import (
  CalculatePerformanceMetrics,
  PlotConfusionMatrix,
  PlotROCAUCCurve,
  HistoryPlotter,
)
from HMB.PyTorchTrainingPipeline import TrainEvaluateClassificationModel

# Enable CuDNN benchmark for optimized convolution algorithms.
torch.backends.cudnn.benchmark = True

def PlotClassHistograms(
  dataLoaders: Dict[str, DataLoader],
  classNames: List[str],
  outputDir: str = "Results",
) -> None:
  # Initialize the dictionary for class counts.
  classCounts = {}
  # Initialize the dictionary for total split counts.
  splitTotals = {}
  # Iterate through each split in the data loaders.
  for splitName, loader in dataLoaders.items():
    # Access the labels directly from the dataset to avoid loading images from disk.
    labelsArr = numpy.array(loader.dataset.labels)
    # Count occurrences of each class.
    counts = numpy.bincount(labelsArr, minlength=len(classNames))
    # Store the counts for the split.
    classCounts[splitName] = counts
    # Store the total count for the split.
    splitTotals[splitName] = len(labelsArr)
  # Determine the number of classes.
  numClasses = len(classNames)
  # Determine the number of splits.
  numSplits = len(classCounts)
  # Check if there are no classes.
  if (numClasses == 0):
    # Return early if no classes.
    return
  # Create a figure for the histograms.
  plt.figure(figsize=(max(6, numClasses * 0.8), 5))
  # Define the width of the bars.
  barWidth = 0.25
  # Define the positions for the bars.
  r1 = numpy.arange(numClasses)
  # Initialize the index for splits.
  splitIdx = 0
  # Initialize the list to store bar containers for labeling.
  barContainers = []
  # Iterate through each split.
  for splitName, counts in classCounts.items():
    # Calculate the position for the current split.
    rPos = [x + splitIdx * barWidth for x in r1]
    # Get the total count for the current split.
    totalCount = splitTotals.get(splitName, 0)
    # Create the label with the count.
    labelWithCount = f"{splitName} (N={totalCount})"
    # Plot the bar chart for the current split and store the container.
    bars = plt.bar(rPos, counts, width=barWidth, edgecolor="grey", label=labelWithCount)
    # Append the bars to the list.
    barContainers.append(bars)
    # Increment the split index.
    splitIdx += 1
  # Set the x-axis ticks to the center of the groups.
  plt.xticks(
    [r + barWidth for r in range(numClasses)],
    classNames,
    rotation=45,
    ha="right",
  )
  # Set the x-axis label.
  plt.xlabel("Class", fontsize=12, fontweight="bold")
  # Set the y-axis label.
  plt.ylabel("Count", fontsize=12, fontweight="bold")
  # Set the plot title.
  plt.title("Class Distribution Across Splits", fontsize=14, fontweight="bold")
  # Add the legend.
  plt.legend()
  # Add count labels above each bar.
  for bars in barContainers:
    # Iterate through each bar in the container.
    for bar in bars:
      # Get the height of the bar.
      yVal = bar.get_height()
      # Check if the value is greater than 0 to avoid clutter.
      if (yVal > 0):
        # Add the text label above the bar.
        plt.text(
          bar.get_x() + bar.get_width() / 2.0,
          yVal,
          int(yVal),
          ha="center",
          va="bottom",
          fontsize=8,
        )
  # Adjust the layout.
  plt.tight_layout()
  # Construct the full file path for the histogram plot.
  histogramFilePath = str(Path(outputDir) / "ClassHistograms.png")
  # Save the figure.
  plt.savefig(histogramFilePath, dpi=720, bbox_inches="tight")
  # Close the plot.
  plt.close()
  # Print the confirmation message.
  fprint(f"Class histograms saved to {histogramFilePath}")


# Define helper function for Background Removal using Otsu's Thresholding.
def RemoveBackground(imgArray, thresholdFactor=0.8):
  '''
  Masks out white background using Otsu's thresholding on grayscale conversion.
  Input: numpy array (H, W, C) in RGB format, float32 [0, 1].
  Output: numpy array (H, W, C) with background set to 0 (black).
  '''
  # Check if image is grayscale.
  if (imgArray.ndim == 2):
    # Apply Otsu directly.
    gray = imgArray
  elif (imgArray.ndim == 3 and imgArray.shape[2] >= 3):
    # Convert to grayscale using standard luminance formula.
    gray = 0.2989 * imgArray[..., 0] + 0.5870 * imgArray[..., 1] + 0.1140 * imgArray[..., 2]
  else:
    # Return original if unsupported.
    return imgArray

  # Scale to uint8 for OpenCV Otsu.
  grayUint8 = (gray * 255).astype(numpy.uint8)

  # Apply Otsu's thresholding.
  _, thresh = cv2.threshold(grayUint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # Create a mask where tissue is present (thresh > 0 means foreground in inverted logic usually,
  # but Otsu on white background will make background 255. We want to keep non-background).
  # Actually, Otsu on white background typically sets background to 255.
  # We want to keep pixels that are NOT background.
  # Let's invert: if pixel is 255 (white/bg), mask it out.
  mask = (thresh < 128).astype(numpy.float32)  # Adjust threshold logic based on data

  # Expand mask to match channels.
  if (imgArray.ndim == 3):
    mask = numpy.expand_dims(mask, axis=-1)

  # Apply mask.
  result = imgArray * mask

  # Return result.
  return result


# Define the function to load pretrained ImageNet weights into custom ViT models.
def LoadPretrainedViTWeights(
  model: torch.nn.Module,
  pretrainedModelName: str = "vit_base_patch16_224",
  numClasses: int = 2,
  verbose: bool = True,
) -> torch.nn.Module:
  # Load the pretrained ViT model from timm with ImageNet weights.
  pretrainedModel = timm.create_model(pretrainedModelName, pretrained=True, num_classes=numClasses)
  # Initialize a counter for successfully transferred layers.
  transferredCount = 0
  # Initialize a counter for skipped layers.
  skippedCount = 0
  # Extract the pretrained state dictionary.
  pretrainedDict = pretrainedModel.state_dict()
  # Extract the custom model state dictionary.
  modelDict = model.state_dict()
  # Define the mapping from pretrained ViT keys to custom model keys.
  keyMapping = {
    # Map the patch embedding projection weights.
    "patch_embed.proj.weight": "patchEmbed.weight",
    # Map the patch embedding projection bias.
    "patch_embed.proj.bias"  : "patchEmbed.bias",
    # Map the class token parameter.
    "cls_token"              : "clsToken",
    # Map the final layer normalization weights.
    "norm.weight"            : "norm.weight",
    # Map the final layer normalization bias.
    "norm.bias"              : "norm.bias",
  }
  # Iterate through the key mapping to transfer compatible weights.
  for pretrainedKey, customKey in keyMapping.items():
    # Check if both keys exist in their respective state dictionaries.
    if (pretrainedKey in pretrainedDict and customKey in modelDict):
      # Check if the tensor shapes match exactly.
      if (pretrainedDict[pretrainedKey].shape == modelDict[customKey].shape):
        # Copy the pretrained weights into the custom model.
        modelDict[customKey] = pretrainedDict[pretrainedKey]
        # Increment the transferred counter.
        transferredCount += 1
      else:
        # Print a shape mismatch warning if verbose mode is enabled.
        if (verbose):
          fprint(
            f"  Shape mismatch for {customKey}: pretrained {pretrainedDict[pretrainedKey].shape} vs custom {modelDict[customKey].shape}. Skipping.")
        # Increment the skipped counter.
        skippedCount += 1
    else:
      # Increment the skipped counter for missing keys.
      skippedCount += 1
  # Handle positional embedding transfer with interpolation if needed.
  if ("pos_embed" in pretrainedDict and "posEmbed" in modelDict):
    # Get the pretrained positional embedding tensor.
    pretrainedPos = pretrainedDict["pos_embed"]
    # Get the custom model positional embedding tensor.
    customPos = modelDict["posEmbed"]
    # Check if the shapes match exactly.
    if (pretrainedPos.shape == customPos.shape):
      # Copy the positional embedding directly.
      modelDict["posEmbed"] = pretrainedPos
      # Increment the transferred counter.
      transferredCount += 1
    else:
      # Interpolate the positional embedding to match the custom model sequence length.
      # Separate the class token position from the spatial positions.
      pretrainedClsPos = pretrainedPos[:, :1, :]
      # Extract the spatial positional embeddings.
      pretrainedSpatialPos = pretrainedPos[:, 1:, :]
      # Get the target spatial sequence length from the custom model.
      targetSpatialLen = customPos.shape[1] - 1
      # Get the source spatial sequence length from the pretrained model.
      sourceSpatialLen = pretrainedSpatialPos.shape[1]
      # Calculate the source grid size assuming a square layout.
      sourceGridSize = int(sourceSpatialLen ** 0.5)
      # Calculate the target grid size assuming a square layout.
      targetGridSize = int(targetSpatialLen ** 0.5)
      # Check if both grid sizes are valid perfect squares.
      if (sourceGridSize * sourceGridSize == sourceSpatialLen and targetGridSize * targetGridSize == targetSpatialLen):
        # Reshape the spatial positions to a 2D grid for interpolation.
        posGrid = pretrainedSpatialPos.reshape(1, sourceGridSize, sourceGridSize, -1).permute(0, 3, 1, 2)
        # Apply bilinear interpolation to resize the positional grid.
        posGridResized = torch.nn.functional.interpolate(
          posGrid,
          size=(targetGridSize, targetGridSize),
          mode="bilinear",
          align_corners=False,
        )
        # Reshape back to sequence format.
        posGridResized = posGridResized.permute(0, 2, 3, 1).reshape(1, targetSpatialLen, -1)
        # Concatenate the class token position with the resized spatial positions.
        modelDict["posEmbed"] = torch.cat([pretrainedClsPos, posGridResized], dim=1)
        # Increment the transferred counter.
        transferredCount += 1
        # Print the interpolation message if verbose mode is enabled.
        if (verbose):
          fprint(f"  Interpolated positional embedding from {sourceSpatialLen} to {targetSpatialLen} spatial tokens.")
      else:
        # Print a warning if grid sizes are not valid.
        if (verbose):
          fprint(f"  Cannot interpolate positional embedding: non-square grid. Skipping.")
        # Increment the skipped counter.
        skippedCount += 1
  # Handle classification head transfer if the number of classes matches.
  if ("head.weight" in pretrainedDict and "head.weight" in modelDict):
    # Check if the head dimensions match.
    if (pretrainedDict["head.weight"].shape == modelDict["head.weight"].shape):
      # Copy the classification head weights.
      modelDict["head.weight"] = pretrainedDict["head.weight"]
      # Copy the classification head bias if it exists.
      if ("head.bias" in pretrainedDict and "head.bias" in modelDict):
        modelDict["head.bias"] = pretrainedDict["head.bias"]
      # Increment the transferred counter.
      transferredCount += 1
  # Load the modified state dictionary back into the custom model.
  model.load_state_dict(modelDict)
  # Print the transfer summary if verbose mode is enabled.
  if (verbose):
    fprint(
      f"  Pretrained weight transfer complete: {transferredCount} layers transferred, {skippedCount} layers skipped.")
    fprint(f"  Custom blocks (KAN/ODE/Spiking/etc.) remain randomly initialized and will learn during fine-tuning.")
  # Return the model with transferred weights.
  return model


class FeatureExtractorWrapper(torch.nn.Module):
  # Initialize the feature extractor wrapper.
  def __init__(self, model):
    # Call the parent neural network module constructor.
    super(FeatureExtractorWrapper, self).__init__()
    # Store the underlying classification model.
    self.model = model
    # Initialize the hook handle.
    self.hookHandle = None
    # Initialize the extracted features storage.
    self.extractedFeatures = None

  # Define the hook function to capture features.
  def _hookFn(self, module, input, output):
    # Store the input of the head module, which represents the penultimate features.
    self.extractedFeatures = input[0]

  # Define the forward pass to extract both logits and penultimate features.
  def forward(self, x):
    # Check if the model is a CLIP classifier which only has a fully connected layer.
    if (isinstance(self.model, CLIPClassifier)):
      # Return the logits and the input features directly for CLIP.
      return self.model(x), x

    # Identify the classification head attribute dynamically.
    headAttr = next((attr for attr in ["head", "mlpHead", "fc", "classifier"] if hasattr(self.model, attr)), None)

    # Check if a valid head attribute was found.
    if (headAttr is not None):
      # Get the head module.
      headModule = getattr(self.model, headAttr)
      # Register a forward hook on the head module to capture its input.
      self.hookHandle = headModule.register_forward_hook(self._hookFn)
      # Perform the standard forward pass.
      logits = self.model(x)
      # Remove the hook after the forward pass.
      if (self.hookHandle is not None):
        # Remove the registered hook.
        self.hookHandle.remove()
        # Reset the hook handle to None.
        self.hookHandle = None
      # Retrieve the captured features.
      features = self.extractedFeatures
      # Check if the extracted features have a sequence dimension.
      if (features.dim() == 3):
        # Average the features over the sequence dimension to obtain a single vector per sample.
        features = features.mean(dim=1)
      # Check if the extracted features have spatial dimensions (e.g., from CNNs like ConvNeXt).
      elif (features.dim() == 4):
        # Average the features over the spatial dimensions to obtain a single vector per sample.
        features = features.mean(dim=(2, 3))
      # Return both the logits and the extracted features.
      return logits, features
    # Fallback for models without a standard head attribute.
    else:
      # Pass the input through the model normally.
      logits = self.model(x)
      # Return the logits and a dummy zero tensor for features.
      return logits, torch.zeros(logits.size(0), 1, device=logits.device)


def LoadConfig(
  configPath: str,
) -> Dict[str, Any]:
  '''
  Load configuration from a YAML or JSON file, or generate a default one.
  '''
  # Check if the file exists.
  if (not Path(configPath).exists()):
    # Raise an error if the file is not found.
    raise FileNotFoundError(f"Configuration file not found: {configPath}")
  # Open the file for reading.
  with open(configPath, "r") as f:
    # Check the file extension.
    if (configPath.endswith(".json")):
      # Load the JSON configuration.
      config = json.load(f)
    elif (configPath.endswith(".yaml") or configPath.endswith(".yml")):
      # Load the YAML configuration.
      config = yaml.safe_load(f)
    else:
      # Raise an error for unsupported file formats.
      raise ValueError("Unsupported configuration file format. Use .json, .yaml, or .yml.")
  # Return the loaded configuration dictionary.
  return config


class GenericImageDataset(Dataset):
  '''
  Custom PyTorch Dataset for generic image classification.
  '''

  # Initialize the generic image dataset loader.
  def __init__(
    self,
    rootDir: str,
    split: str,
    transform: transforms.Compose,
    imageSize: int,
    useBackgroundRemoval: bool = False,
    useColorDeconvolution: bool = False,
  ) -> None:
    # Call the parent class constructor.
    super().__init__()
    # Store the root directory path.
    self.rootDir = Path(rootDir)
    # Store the dataset split name.
    self.split = split
    # Store the transform function.
    self.transform = transform
    # Store the target image size.
    self.imageSize = imageSize
    # Store background removal flag.
    self.useBackgroundRemoval = useBackgroundRemoval
    # Store color deconvolution flag.
    self.useColorDeconvolution = useColorDeconvolution
    # Initialize the list for image paths.
    self.imagePaths = []
    # Initialize the list for labels.
    self.labels = []
    # Initialize the list for filenames.
    self.filenames = []
    # Initialize the list for sample weights.
    self.weights = []
    # Build the dataset index.
    self._buildIndex()
    # Calculate class weights for sampling.
    self._calculateWeights()

  # Scan the dataset directory and populate image paths and labels lists.
  def _buildIndex(self) -> None:
    # Determine the target split directory.
    targetDir = self.rootDir / self.split
    # Get sorted list of class directories.
    classDirs = sorted([d for d in targetDir.iterdir() if d.is_dir()])
    # Iterate through each class directory.
    for classIdx, classDir in enumerate(classDirs):
      # Recursively find all image files in the class directory.
      for imgPath in classDir.rglob("*"):
        # Filter by common image extensions.
        if (imgPath.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]):
          # Add the valid image path.
          self.imagePaths.append(str(imgPath))
          # Add the corresponding label.
          self.labels.append(classIdx)
          # Add the filename.
          self.filenames.append(imgPath.name)
    # Print the number of loaded images.
    fprint(f"Loaded {len(self.imagePaths)} images from {len(classDirs)} classes in \"{self.split}\" split.")

  # Calculate inverse frequency weights for each sample.
  def _calculateWeights(self) -> None:
    # Check if labels exist.
    if (len(self.labels) == 0):
      # Return early if no labels.
      return
    # Count occurrences of each class.
    classCounts = numpy.bincount(self.labels)
    # Calculate total samples.
    totalSamples = len(self.labels)
    # Calculate weight for each class (inverse frequency).
    classWeights = totalSamples / (len(classCounts) * classCounts + 1e-6)
    # Assign weight to each sample based on its label.
    self.weights = [classWeights[label] for label in self.labels]

  # Return the total number of samples in the dataset.
  def __len__(self) -> int:
    # Return the length of the image paths list.
    return len(self.imagePaths)

  # Load and return a single image-label pair.
  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
    # Get the image file path from the list.
    imgPath = self.imagePaths[idx]
    # Check the file extension to determine the loading method.
    if (imgPath.lower().endswith((".tif", ".tiff"))):
      # Load multi-channel TIFF images using tifffile to preserve all channels.
      imgArray = tifffile.imread(imgPath)
      # Check if the TIFF was saved in (C, H, W) format.
      if (imgArray.ndim == 3 and imgArray.shape[0] < imgArray.shape[1] and imgArray.shape[0] < imgArray.shape[2]):
        # Transpose from (C, H, W) to (H, W, C) for consistent processing.
        imgArray = numpy.transpose(imgArray, (1, 2, 0))
    else:
      # Load standard images using OpenCV without converting the color space.
      imgArray = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
      # Convert from BGR to RGB if the image has 3 or 4 channels.
      if (imgArray.ndim == 3 and imgArray.shape[2] >= 3):
        # Reverse the channel order for RGB compatibility.
        imgArray = imgArray[..., ::-1]

    # Standardize the number of channels to 3 (RGB) to prevent DataLoader stacking errors.
    # This handles mixed datasets containing RGB, RGBA, and grayscale images.
    if (imgArray.ndim == 2):
      # Convert grayscale (H, W) to 3 channels (H, W, 3).
      imgArray = numpy.stack((imgArray, imgArray, imgArray), axis=-1)
    elif (imgArray.ndim == 3):
      # Check if the image has exactly 1 channel.
      if (imgArray.shape[2] == 1):
        # Convert 1 channel (H, W, 1) to 3 channels (H, W, 3).
        imgArray = numpy.concatenate([imgArray, imgArray, imgArray], axis=-1)
      # Check if the image has exactly 4 channels (RGBA).
      elif (imgArray.shape[2] == 4):
        # Convert 4 channels to 3 channels by dropping the alpha channel.
        imgArray = imgArray[..., :3]
      # Check if the image has more than 4 channels.
      elif (imgArray.shape[2] > 4):
        # Keep only the first 3 channels for images with more than 4 channels.
        imgArray = imgArray[..., :3]

    # Check if the image needs to be converted to float and normalized.
    if (imgArray.dtype == numpy.uint8):
      # Normalize uint8 images to the [0.0, 1.0] range.
      imgArray = imgArray.astype(numpy.float32) / 255.0
    elif (imgArray.dtype != numpy.float32):
      # Convert other types to float32.
      imgArray = imgArray.astype(numpy.float32)

    # Retrieve the corresponding label for this image.
    label = self.labels[idx]
    # Get the filename of the image.
    filename = self.filenames[idx]

    # Apply transforms if they have been specified.
    if (self.transform is not None):
      # Convert numpy array to PIL Image for torchvision transforms.
      if (imgArray.dtype == numpy.float32):
        # Convert float [0,1] to uint8 [0,255] for PIL.
        imgPIL = Image.fromarray((imgArray * 255).astype(numpy.uint8))
      else:
        # Convert directly to PIL Image.
        imgPIL = Image.fromarray(imgArray)

      # Apply the transform chain.
      imgOutput = self.transform(imgPIL)
    else:
      # No transform, convert manually to Tensor.
      imgOutput = torch.from_numpy(imgArray).permute(2, 0, 1)

    # Return the processed image and label as a tuple.
    return imgOutput, label


class PyTorchFolderBasedDataPipeline:
  '''
  A unified pipeline to handle folder scanning, auto-splitting, and PyTorch DataLoader creation.
  '''

  # Initialize the data pipeline.
  def __init__(
    self,
    dataDir: str,
    batchSize: int = 32,
    imageSize: int = 224,
    numWorkers: int = 1,
    useAugmentation: bool = True,
    useStainJitter: bool = False,
    useBackgroundRemoval: bool = False,
    useColorDeconvolution: bool = False,
  ) -> None:
    # Store the root data directory.
    self.dataDir = Path(dataDir)
    # Store the batch size.
    self.batchSize = batchSize
    # Store the target image size.
    self.imageSize = imageSize
    # Store the number of workers.
    self.numWorkers = numWorkers
    # Store the augmentation flag.
    self.useAugmentation = useAugmentation
    # Store stain jitter flag.
    self.useStainJitter = useStainJitter
    # Store background removal flag.
    self.useBackgroundRemoval = useBackgroundRemoval
    # Store color deconvolution flag.
    self.useColorDeconvolution = useColorDeconvolution
    # Initialize the dictionary for dataset lengths.
    self.lengths = {}
    # Initialize the list for class names.
    self.classNames = []
    # Prepare the directory splits.
    self.rootPath = self._prepareSplits()
    # Build the dataset index to discover classes and calculate lengths.
    self._buildIndex()
    # Define the training transforms.
    self.trainTransform = self._getTrainTransform()
    # Define the validation and test transforms.
    self.valTestTransform = self._getValTestTransform()
    # Build the training pipeline.
    self.train = self._buildPipeline("train", isTraining=True)
    # Build the validation pipeline.
    self.val = self._buildPipeline("val", isTraining=False)
    # Build the test pipeline.
    self.test = self._buildPipeline("test", isTraining=False)

  # Check for train/val/test folders and auto-split if necessary.
  def _prepareSplits(self) -> Path:
    # Define the root path.
    rootPath = self.dataDir
    # Check if the train directory exists.
    if ((rootPath / "train").exists()):
      # Return the original root path.
      return rootPath
    # Check if there are class directories in the root path to split.
    classDirs = [
      d for d in rootPath.iterdir()
      if (d.is_dir() and d.name not in ["train", "val", "test", "SplitDataset"])
    ]
    # Check if class directories were found.
    if (len(classDirs) > 0):
      fprint("Train directory not found. Using split-folders to create train/val/test subsets...")
      # Import the splitfolders module dynamically.
      import splitfolders
      # Define the output directory for the split dataset.
      rootParent = rootPath.parent
      # Define the split output directory path.
      splitOutputDir = rootParent / "SplitDataset"
      # Check if the split directory does not exist.
      if (not splitOutputDir.exists()):
        # Use splitfolders to create the train/val/test subsets.
        splitfolders.ratio(input=str(rootPath), output=str(splitOutputDir), seed=1337, ratio=(0.8, 0.1, 0.1))
      # Return the new split directory path.
      return splitOutputDir
    else:
      # Raise an error if train directory is missing and no class directories are found.
      raise FileNotFoundError("Train directory not found and no class directories found to split.")

  # Scan the directories to discover class names and calculate split lengths.
  def _buildIndex(self) -> None:
    # Determine the target train directory.
    targetDir = self.rootPath / "train"
    # Get sorted list of class directories.
    classDirs = sorted([d for d in targetDir.iterdir() if (d.is_dir())])
    # Store the class names.
    self.classNames = [d.name for d in classDirs]
    fprint(f"Found {len(classDirs)} classes: {self.classNames}")
    # Iterate through train, val, and test splits to calculate lengths.
    for split in ["train", "val", "test"]:
      # Determine the split directory.
      splitDir = self.rootPath / split
      # Check if the split directory exists.
      if (splitDir.exists()):
        # Initialize the count for the split.
        count = 0
        # Iterate through class directories in the split.
        for classDir in classDirs:
          # Determine the class directory in the split.
          splitClassDir = splitDir / classDir.name
          # Check if the class directory exists in the split.
          if (splitClassDir.exists()):
            # Count the valid image files.
            for imgPath in splitClassDir.rglob("*"):
              # Filter by common image extensions.
              if (imgPath.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]):
                # Increment the count.
                count += 1
        # Store the length for the split.
        self.lengths[split.capitalize()] = count
        # Print the number of samples for the split.
        fprint(f"{split.capitalize()} samples: {count}")

  # Define the training transforms with data augmentation.
  def _getTrainTransform(self) -> transforms.Compose:
    if (not self.useAugmentation):
      # If augmentation is disabled, return only resizing and ToTensor.
      return transforms.Compose([
        transforms.Resize((self.imageSize, self.imageSize)),
        transforms.ToTensor()
      ])
    # Return the composed training transforms.
    transList = [
      # Resize the image slightly larger for random crop.
      transforms.Resize((int(self.imageSize * 1.1), int(self.imageSize * 1.1))),
      # Apply random resized cropping.
      transforms.RandomCrop(self.imageSize),
      # Apply random horizontal flipping.
      transforms.RandomHorizontalFlip(),
      # Apply random vertical flipping.
      transforms.RandomVerticalFlip(),
      # Apply random rotation for histopathology invariance.
      transforms.RandomRotation(360),
    ]
    if (self.useStainJitter):
      # Apply stain jittering if enabled.
      transList.append(StainJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    # Convert to Tensor and normalize to [0, 1].
    transList.append(transforms.ToTensor())
    return transforms.Compose(transList)

  # Define the validation and test transforms without augmentation.
  def _getValTestTransform(self) -> transforms.Compose:
    # Return the composed validation and test transforms.
    return transforms.Compose([
      # Resize the image to the target size.
      transforms.Resize((self.imageSize, self.imageSize)),
      # Convert to Tensor.
      transforms.ToTensor()
    ])

  # Build the PyTorch DataLoader pipeline for a specific split.
  def _buildPipeline(
    self,
    split: str,
    isTraining: bool,
  ) -> Optional[DataLoader]:
    # Determine the target split directory.
    splitDir = self.rootPath / split
    # Check if the split directory exists.
    if (not splitDir.exists()):
      # Return None for the missing split.
      return None
    # Select the appropriate transform based on the training flag.
    transform = self.trainTransform if (isTraining) else self.valTestTransform
    # Create the dataset object.
    dataset = GenericImageDataset(
      rootDir=self.rootPath,
      split=split,
      transform=transform,
      imageSize=self.imageSize,
      useBackgroundRemoval=self.useBackgroundRemoval,
      useColorDeconvolution=self.useColorDeconvolution,
    )

    # Initialize sampler.
    sampler = None
    # Check if training and weights exist.
    if (isTraining and hasattr(dataset, "weights") and len(dataset.weights) > 0):
      # Use WeightedRandomSampler for imbalanced histopathology data.
      sampler = WeightedRandomSampler(
        weights=dataset.weights,
        num_samples=len(dataset.weights),
        replacement=True
      )

    # Create and return the DataLoader.
    return DataLoader(
      dataset,
      batch_size=self.batchSize,
      shuffle=(sampler is None and isTraining),
      sampler=sampler,
      num_workers=self.numWorkers,
      pin_memory=True,
      persistent_workers=True if (self.numWorkers > 0) else False,
    )


def AdaptModelInputChannels(
  model: torch.nn.Module,
  targetChannels: int,
) -> torch.nn.Module:
  '''
  Finds the first Conv2D layer in the model and adapts its weights to accept "targetChannels".
  '''
  # Initialize variables to store the layer and its path.
  firstConv = None
  convPath = ""

  # Iterate through the model to find the first Conv2D layer.
  for name, module in model.named_modules():
    # Check if the current module is a Conv2D layer.
    if (isinstance(module, torch.nn.Conv2d)):
      # Store the reference to the layer.
      firstConv = module
      # Store the path to the layer.
      convPath = name
      # Break out of the loop since we only need the first one.
      break

  # Check if a Conv2D layer was found.
  if (firstConv is None):
    # Print a warning if no Conv2D layer is found.
    fprint("Warning: No Conv2D layer found to adapt input channels.")
    # Return the model unmodified.
    return model

  # Get the original number of input channels.
  originalChannels = firstConv.in_channels

  # Check if the channels already match.
  if (originalChannels == targetChannels):
    # Return the model unmodified.
    return model

  # Print the adaptation message.
  fprint(f"Adapting first Conv2D layer from {originalChannels} to {targetChannels} channels...")

  # Get the original weights.
  originalWeights = firstConv.weight.data

  # Check if we need to increase the number of channels.
  if (targetChannels > originalChannels):
    # Calculate the number of repeats needed.
    repeats = (targetChannels + originalChannels - 1) // originalChannels
    # Repeat the weights along the input channel dimension.
    newWeights = originalWeights.repeat(1, repeats, 1, 1)
    # Slice to the exact target channels.
    newWeights = newWeights[:, :targetChannels, :, :]
    # Normalize the weights to maintain the same magnitude of activations.
    newWeights = newWeights * (originalChannels / targetChannels)
  else:
    # Slice the weights to the target channels if reducing.
    newWeights = originalWeights[:, :targetChannels, :, :]

  # Create a new Conv2D layer with the updated input channels.
  newConv = torch.nn.Conv2d(
    in_channels=targetChannels,
    out_channels=firstConv.out_channels,
    kernel_size=firstConv.kernel_size,
    stride=firstConv.stride,
    padding=firstConv.padding,
    dilation=firstConv.dilation,
    groups=firstConv.groups,
    bias=(firstConv.bias is not None),
    padding_mode=firstConv.padding_mode
  )

  # Assign the new weights to the new layer.
  newConv.weight.data = newWeights

  # Check if the original layer had a bias.
  if (firstConv.bias is not None):
    # Copy the bias to the new layer.
    newConv.bias.data = firstConv.bias.data.clone()

  # Split the path to traverse the model hierarchy.
  pathParts = convPath.split(".")
  # Initialize the current module as the root model.
  currentModule = model

  # Traverse to the parent module.
  for part in pathParts[:-1]:
    # Check if the part is a digit (for Sequential or ModuleList).
    if (part.isdigit()):
      # Access by index.
      currentModule = currentModule[int(part)]
    else:
      # Access by attribute.
      currentModule = getattr(currentModule, part)

  # Get the final attribute name.
  finalAttr = pathParts[-1]

  # Replace the old layer with the new layer.
  setattr(currentModule, finalAttr, newConv)

  # Return the modified model.
  return model


def CreateFitViTModel(
  trainLoader: DataLoader,
  valLoader: DataLoader,
  modelName: str,
  numClasses: int,
  numEpochs: int = 50,
  learningRate: float = 1e-4,
  device: str = "cuda",
  outputDir: str = "Results",
  patience: int = 10,
  imageSize: int = 224,
  optimizerName: str = "Adam",
  useAmp: bool = False,
  accumulationSteps: int = 1,
  useEma: bool = False,
  useMixup: bool = False,
  mixupAlpha: float = 0.2,
  cutmixAlpha: float = 1.0,
  lossFunction: str = "CrossEntropy",
  labelSmoothing: float = 0.0,
  schedulerName: str = "None",
  judgeBy: str = "val_accuracy",
  saveEvery: Optional[int] = None,
  maxGradNorm: Optional[float] = None,
  resumeFromCheckpoint: bool = False,
  useTopoWassersteinLoss: bool = True,
  epsilon: float = 0.1,
  lambdaTopo: float = 0.1,
  lambdaContrastive: float = 0.1,
  usePretrainedCustomModels: bool = False,
  pretrainedModelName: str = "vit_base_patch16_224",
) -> Tuple[torch.nn.Module, Dict[str, List[float]], Any, Dict[str, Any]]:
  # Determine effective image size based on model.
  effectiveImageSize = imageSize
  if (modelName == "SwinTransformerV2"):
    effectiveImageSize = 256
  elif (modelName == "SwinTransformerLarge384"):
    effectiveImageSize = 384
  elif (modelName == "EVA02Large"):
    effectiveImageSize = 448

  # Build the ViT model with the specified image size.
  baseModel, secondaryModel = BuildViTModel(
    modelName=modelName,
    numClasses=numClasses,
    device=device,
    imageSize=effectiveImageSize,
    usePretrainedCustomModels=usePretrainedCustomModels,
    pretrainedModelName=pretrainedModelName,
  )

  # Wrap the base model to enable feature extraction for topological loss.
  model = baseModel  # FeatureExtractorWrapper(baseModel)

  # Determine the input channels from the dataset by peeking at one batch.
  sampleInputs, _ = next(iter(trainLoader))
  targetChannels = sampleInputs.shape[1]

  # Adapt the model's first layer to match the dataset's channel count.
  model = AdaptModelInputChannels(model, targetChannels)

  # Move the model to the specified device.
  model = model.to(device)

  # Define the loss function based on the configuration.
  if (lossFunction == "Focal"):
    trainLabels = trainLoader.dataset.labels
    classCounts = numpy.bincount(trainLabels, minlength=numClasses).tolist()
    criterion = FocalLossRobust(numClasses=numClasses, classCounts=classCounts)
  elif (lossFunction == "LabelSmoothing"):
    criterion = LSCE_Loss(labelSmoothing=labelSmoothing)
  else:
    criterion = CrossEntropyLossWrapper()

  # Initialize the Wasserstein Topological Loss module conditionally.
  # Note: TrainEvaluateClassificationModel does not natively support composite losses like WassersteinTopologicalLoss
  # without modification. If strict adherence to HMB pipeline is required, this complex loss logic
  # might need to be wrapped in a custom criterion or the pipeline extended.
  # For now, we assume standard losses for the pipeline replacement.
  # If TopoLoss is critical, we must stick to the manual loop or extend the pipeline.
  # Assuming we want to use the HMB Pipeline, we disable TopoLoss for this refactor or wrap it.
  topoWassersteinCriterion = None
  if (useTopoWassersteinLoss):
    fprint(
      "Warning: Topological Wasserstein Loss is not directly supported by the standard HMB Training Pipeline "
      "wrapper without custom criterion integration. Using standard loss."
    )
    # To strictly use HMB pipeline, we skip TopoLoss here or implement a custom Criterion class that wraps it.

  # Define the model checkpoint path.
  modelCheckpointPath = str(Path(outputDir) / "FinalModel.pt")

  # Define Optimizer.
  opt = GetOptimizer(model, optimizerName, learningRate)

  # Define Scheduler.
  scheduler = None
  if (schedulerName == "CosineWarmup"):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=numEpochs, eta_min=1e-6)
  elif (schedulerName == "ReduceLROnPlateau"):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

  # Use HMB Training Pipeline.
  # Note: This replaces the manual loop, early stopping, and saving logic.
  history = TrainEvaluateClassificationModel(
    model=model,
    criterion=criterion,
    device=torch.device(device),
    bestModelStoragePath=modelCheckpointPath,
    noOfClasses=numClasses,
    numEpochs=numEpochs,
    optimizer=opt,
    scaler=EnableMixedPrecision(model) if useAmp else None,
    scheduler=scheduler,
    trainLoader=trainLoader,
    valLoader=valLoader,
    resumeFromCheckpoint=resumeFromCheckpoint,
    finalModelStoragePath=str(Path(outputDir) / "FinalModel.pt"),
    judgeBy=judgeBy,
    earlyStoppingPatience=patience,
    verbose=True,
    gradAccumSteps=accumulationSteps,
    maxGradNorm=maxGradNorm,
    useAmp=useAmp,
    useMixupFn=useMixup,
    mixUpAlpha=mixupAlpha,
    useEma=useEma,
    saveEvery=saveEvery
  )

  # Convert history keys to match expected format for Plotting if necessary.
  # HMB Pipeline returns "train_loss", "val_loss", etc.
  # We retain the lowercase keys as strictly required by HistoryPlotter and CSV exporters.
  formattedHistory = {
    "train_loss": history.get("train_loss", []),
    "val_loss": history.get("val_loss", []),
    "train_accuracy": history.get("train_accuracy", []),
    "val_accuracy": history.get("val_accuracy", []),
  }

  # Load the best model.
  model = LoadModel(model, modelCheckpointPath, device=device)

  configs = {
    "ModelName"          : str(modelName),
    "NumClasses"         : int(numClasses),
    "NumEpochs"          : int(numEpochs),
    "LearningRate"       : float(learningRate),
    "OptimizerName"      : str(optimizerName),
    "ModelCheckpointPath": str(modelCheckpointPath),
  }

  return model, formattedHistory, secondaryModel, configs


def EvaluateModel(
  model: torch.nn.Module,
  dataLoader: DataLoader,
  outputDir: str = "Results",
  savePlots: bool = True,
  classNames: Optional[List[str]] = None,
  prefix: str = "Test",
  device: str = "cuda",
  numClasses: int = 3,
  secondaryModel: Any = None,
) -> Dict[str, Union[float, numpy.ndarray, str]]:
  # Print the evaluation start message.
  fprint(f"Starting evaluation: savePlots={savePlots} | outputDir={outputDir} | prefix={prefix}")
  # Create a folder for the evaluation results if it doesn't exist for the current prefix.
  evalOutputDir = Path(outputDir) / prefix
  # Create the directory if it doesn't exist.
  evalOutputDir.mkdir(parents=True, exist_ok=True)
  # Initialize the list to collect predictions.
  allPreds = []
  # Initialize the list to collect ground truth labels.
  allLabels = []
  # Initialize the list to collect predicted probabilities (for detailed analysis).
  allPredProbs = []
  # Set the model to evaluation mode.
  model.eval()
  # Disable gradient computation.
  with torch.no_grad():
    # Initialize the progress bar for evaluation.
    progressBar = tqdm(dataLoader, desc="Evaluating", leave=False)
    # Iterate through evaluation batches.
    for inputs, labels in progressBar:
      # Move the inputs to the device.
      inputs = inputs.to(device)
      # Check if using CLIP.
      if (secondaryModel is not None):
        # CLIP strictly requires 3 channels. Slice the input if necessary.
        if (inputs.shape[1] > 3):
          # Extract the first three channels for CLIP.
          clipInputs = inputs[:, :3, :, :]
        else:
          # Use the input as is.
          clipInputs = inputs
        # Extract features using the frozen CLIP visual encoder.
        features = secondaryModel.encode_image(clipInputs).float()
        # Perform the forward pass to obtain model predictions.
        outputsTuple = model(features)
      else:
        # Perform the forward pass to obtain model predictions.
        outputsTuple = model(inputs)

      # Unpack logits from the tuple returned by FeatureExtractorWrapper.
      if (isinstance(outputsTuple, tuple)):
        outputs = outputsTuple[0]
      else:
        outputs = outputsTuple
      # Apply softmax for both binary and multiclass classification (since model outputs [B, numClasses] logits).
      probs = torch.softmax(outputs, dim=1).cpu().numpy()
      # Get the predicted class indices.
      preds = numpy.argmax(probs, axis=1)
      # Append the batch predictions to the accumulation list.
      allPreds.extend(preds)
      # Append the batch labels to the accumulation list.
      allLabels.extend(labels.numpy())
      # Append the batch predicted probabilities to the accumulation list.
      allPredProbs.extend(probs)
  # Convert the accumulated predictions list to a numpy array.
  allPreds = numpy.array(allPreds)
  # Convert the accumulated labels list to a numpy array.
  allLabels = numpy.array(allLabels)
  # Convert predicted probabilities to numpy array.
  allPredProbs = numpy.array(allPredProbs)

  # Generate the confusion matrix for detailed error analysis.
  confMatrix = confusion_matrix(allLabels, allPreds)
  # Check if class names are provided.
  if (classNames is not None):
    # Use the provided class names.
    targetNames = classNames
  else:
    # Generate default class names based on unique labels.
    targetNames = [f"Class{i}" for i in range(len(numpy.unique(allLabels)))]

  # Calculate the performance metrics using HMB Helper (includes Kappa, AUC, etc. if configured).
  pmMetrics = CalculatePerformanceMetrics(confMatrix, addWeightedAverage=True, addPerClass=True)

  # Calculate AUC-ROC and AUPRC if there are more than 1 class.
  try:
    # For multiclass, use "ovr" (One-vs-Rest) or "ovo" (One-vs-One). "ovr" is standard.
    # average="macro" calculates the mean of labels.
    aucRoc = roc_auc_score(allLabels, allPredProbs, multi_class="ovr", average="macro")
    auprc = average_precision_score(allLabels, allPredProbs, average="macro")

    pmMetrics["AUC_ROC_Macro"] = aucRoc
    pmMetrics["AUPRC_Macro"] = auprc

    # Plot ROC Curves if requested.
    if (savePlots):
      filePath = str(evalOutputDir / f"{prefix}_ROC.png")
      PlotROCAUCCurve(
        yTrue=allLabels,
        yPred=allPredProbs,
        classes=targetNames,
        areProbabilities=True,
        title="Receiver Operating Characteristic (ROC) Curve",
        figSize=(10, 8),
        display=False,
        save=True,
        fileName=filePath,
        fontSize=15,
        plotDiagonal=True,
        annotateAUC=True,
        showLegend=True,
        returnFig=False,
        dpi=720,
      )
      fprint(f"ROC curves saved to {filePath}")

  except ValueError:
    # Handle cases where AUC cannot be computed (e.g., only one class present in batch).
    pmMetrics["AUC_ROC_Macro"] = "N/A"
    pmMetrics["AUPRC_Macro"] = "N/A"

  # Generate the detailed classification report with per-class metrics.
  classReport = classification_report(
    allLabels, allPreds, target_names=targetNames, zero_division=0
  )
  # Save the confusion matrix plot if visualization is requested.
  if (savePlots):
    # Call the function to plot the confusion matrix directly.
    filePath = str(evalOutputDir / f"{prefix}CM.png")
    PlotConfusionMatrix(
      cm=confMatrix,
      classes=targetNames,
      normalize=False,
      title="Confusion Matrix",
      display=False,
      save=True,
      fileName=filePath,
      fontSize=12,
      annotate=True,
      figSize=(8, 6),
      colorbar=True,
      returnFig=False,
      dpi=720,
    )
    fprint(f"Confusion matrix saved to {filePath}")

  # Save the confusion matrix to a CSV file with class names only in the first row and first column.
  with open(Path(evalOutputDir) / f"{prefix}CM.csv", "w", newline="") as csvFile:
    # Create a CSV writer object.
    csvWriter = csv.writer(csvFile)
    # Write the first row with an empty top-left corner and the class names as column headers.
    csvWriter.writerow([""] + targetNames)
    # Iterate through the confusion matrix rows.
    for i, className in enumerate(targetNames):
      # Write the row with the class name in the first column (row header) followed by the matrix values.
      csvWriter.writerow([className] + confMatrix[i].tolist())
  # Save the classification report to a text file for reference.
  if (savePlots):
    # Open the classification report file in write mode.
    with open(Path(evalOutputDir) / f"{prefix}ClassificationReport.txt", "w") as f:
      # Write the classification report to the file.
      f.write(classReport)
  # Compile all metrics into a structured dictionary for return.
  metrics = {
    **pmMetrics,
    "ConfusionMatrix"     : confMatrix.tolist(),
    "ClassificationReport": classReport,
    "Predictions"         : allPreds.tolist(),
    "Labels"              : allLabels.tolist(),
  }
  # Store the metrics in a JSON file for downstream use.
  with open(Path(evalOutputDir) / f"{prefix}EvaluationMetrics.json", "w") as jsonFile:
    # Dump the metrics dictionary to the JSON file.
    json.dump(metrics, jsonFile, indent=4, cls=NumpyEncoder)
  # Store detailed predictions by appending to a CSV file for detailed analysis.
  with open(Path(evalOutputDir) / f"{prefix}DetailedPredictions.csv", "w", newline="") as csvFile:
    # Create a CSV writer object.
    csvWriter = csv.writer(csvFile)
    # Write the header row for the detailed predictions CSV.
    csvWriter.writerow(["Split", "Index", "Actual", "Pred", "PredProb"])
    # Iterate through the predictions, labels, and predicted probabilities.
    for idx, (actual, pred, predProb) in enumerate(zip(allLabels, allPreds, allPredProbs)):
      # Write a row for each prediction with the split, index, actual label, predicted label, and predicted probabilities.
      csvWriter.writerow(
        [prefix, idx, actual, pred, predProb.tolist() if isinstance(predProb, numpy.ndarray) else predProb]
      )


def RunCompletePipeline(
  dataDir: str,
  outputDir: str = "Results",
  modelName: str = "StandardViT",
  numEpochs: int = 50,
  batchSize: int = 16,
  imageSize: int = 224,
  learningRate: float = 1e-4,
  devicePref: Optional[str] = "auto",
  patience: int = 10,
  optimizerName: str = "Adam",
  useAmp: bool = False,
  accumulationSteps: int = 1,
  useEma: bool = False,
  useMixup: bool = False,
  mixupAlpha: float = 0.2,
  cutmixAlpha: float = 1.0,
  lossFunction: str = "CrossEntropy",
  labelSmoothing: float = 0.0,
  schedulerName: str = "None",
  useAugmentation: bool = True,
  useStainJitter: bool = False,
  useBackgroundRemoval: bool = False,
  useColorDeconvolution: bool = False,
  useTopoWassersteinLoss: bool = True,
  epsilon: float = 0.1,
  lambdaTopo: float = 0.1,
  lambdaContrastive: float = 0.1,
  usePretrainedCustomModels: bool = False,
  pretrainedModelName: str = "vit_base_patch16_224",
) -> Dict[str, Any]:
  # Print the pipeline initialization message.
  fprint("Starting ViT Image Classification Pipeline")
  # Determine the computation device.
  if (devicePref == "auto"):
    # Use CUDA if available, otherwise CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
  else:
    # Use the specified device.
    device = devicePref
  # Print the selected device.
  fprint(f"Using device: {device}")
  # Print the data directory.
  fprint(f"Data Directory: {dataDir}")
  # Print the output directory.
  fprint(f"Output Directory: {outputDir}")
  # Create the output directory structure for organized results.
  Path(outputDir).mkdir(parents=True, exist_ok=True)

  # Determine effective image size based on model.
  effectiveImageSize = imageSize
  if (modelName == "SwinTransformerV2"):
    effectiveImageSize = 256
    fprint(f"Detected SwinTransformerV2. Overriding image size to {effectiveImageSize}.")

  # Generate data loaders using the unified pipeline.
  fprint("Preparing data loaders...")
  # Instantiate the data pipeline with the EFFECTIVE image size.
  pipeline = PyTorchFolderBasedDataPipeline(
    dataDir=dataDir,
    batchSize=batchSize,
    imageSize=effectiveImageSize,
    useAugmentation=useAugmentation,
    useStainJitter=useStainJitter,
    useBackgroundRemoval=useBackgroundRemoval,
    useColorDeconvolution=useColorDeconvolution,
  )
  # Print the data loaders ready message.
  fprint("Data loaders ready.")
  # Collect available loaders into a dictionary.
  availableLoaders = {}
  # Check if the training loader exists.
  if (pipeline.train):
    # Add the training loader to the dictionary.
    availableLoaders["Train"] = pipeline.train
  # Check if the validation loader exists.
  if (pipeline.val):
    # Add the validation loader to the dictionary.
    availableLoaders["Val"] = pipeline.val
  # Handle the case where the validation loader is missing.
  else:
    # Print a warning about the missing validation loader.
    fprint("Warning: Validation loader not found. Using testing loader for validation.")
    # Assign the test loader to the validation loader variable.
    pipeline.val = pipeline.test
    # Add the assigned loader to the dictionary.
    availableLoaders["Val"] = pipeline.val
  # Check if the test loader exists.
  if (pipeline.test):
    # Add the test loader to the dictionary.
    availableLoaders["Test"] = pipeline.test
  # Handle the case where the test loader is missing.
  else:
    # Check if the validation loader exists.
    if (pipeline.val):
      # Print a warning about the missing test loader.
      fprint("Warning: Test loader not found. Using validation loader for testing.")
      # Assign the validation loader to the test loader variable.
      pipeline.test = pipeline.val
      # Add the assigned loader to the dictionary.
      availableLoaders["Test"] = pipeline.test
    # Handle the case where both test and validation loaders are missing.
    else:
      # Print a warning about missing test and validation loaders.
      fprint("Warning: Test loader not found. Using training loader for testing.")
      # Assign the training loader to the test loader variable.
      pipeline.test = pipeline.train
      # Add the assigned loader to the dictionary.
      availableLoaders["Test"] = pipeline.test
  # Ensure the validation loader is not None if both were initially missing.
  if (pipeline.val is None):
    # Assign the test loader to the validation loader variable.
    pipeline.val = pipeline.test
    # Add the assigned loader to the dictionary.
    availableLoaders["Val"] = pipeline.val
  # Plot class histograms for all splits.
  PlotClassHistograms(
    dataLoaders=availableLoaders,
    classNames=pipeline.classNames,
    outputDir=outputDir,
  )
  # Execute the training process.
  trainedModel, history, secondaryModel, configs = CreateFitViTModel(
    trainLoader=pipeline.train,
    valLoader=pipeline.val,
    modelName=modelName,
    numClasses=len(pipeline.classNames),
    numEpochs=numEpochs,
    learningRate=learningRate,
    device=device,
    outputDir=outputDir,
    patience=patience,
    imageSize=imageSize,
    optimizerName=optimizerName,
    useAmp=useAmp,
    accumulationSteps=accumulationSteps,
    useEma=useEma,
    useMixup=useMixup,
    mixupAlpha=mixupAlpha,
    cutmixAlpha=cutmixAlpha,
    lossFunction=lossFunction,
    labelSmoothing=labelSmoothing,
    schedulerName=schedulerName,
    useTopoWassersteinLoss=useTopoWassersteinLoss,
    epsilon=epsilon,
    lambdaTopo=lambdaTopo,
    lambdaContrastive=lambdaContrastive,
    usePretrainedCustomModels=usePretrainedCustomModels,
    pretrainedModelName=pretrainedModelName,
  )
  print(f"Training complete. Best model saved to {configs['ModelCheckpointPath']}")
  # Generate and save training history visualization plots using HMB Helper.
  HistoryPlotter(
    history=history,
    title="Training History",
    metrics=("loss", "accuracy"),
    xLabel="Epochs",
    fontSize=14,
    save=True,
    savePath=str(Path(outputDir) / "TrainingHistory.pdf"),
    dpi=720,
    display=False,
    figSize=(14, 5),
    returnFig=False,
    smooth=True,
  )
  fprint(f"Training history plots saved to {outputDir}/TrainingHistory.pdf")

  # Load the best model from the checkpoint for evaluation.
  trainedModel = LoadModel(trainedModel, filename=configs["ModelCheckpointPath"], device=device)
  # # Load the best model from the checkpoint for evaluation.
  # trainedModel.load_state_dict(torch.load(configs["ModelCheckpointPath"]))
  # # Move the model to the device.
  # trainedModel = trainedModel.to(device)

  # Evaluate the model on available splits and collect metrics.
  for loader, prefix in zip(
    [pipeline.test, pipeline.val, pipeline.train],
    ["Test", "Val", "Train"],
  ):
    # Check if the loader is not None.
    if (loader is not None):
      # Perform evaluation.
      EvaluateModel(
        model=trainedModel,
        dataLoader=loader,
        outputDir=outputDir,
        savePlots=True,
        classNames=pipeline.classNames,
        prefix=prefix,
        device=device,
        numClasses=len(pipeline.classNames),
        secondaryModel=secondaryModel,
      )
  # Print the pipeline completion summary for user confirmation.
  fprint(f"Pipeline Complete | Results saved to {outputDir}")


if (__name__ == "__main__"):
  # Check if CUDA is available and set the device accordingly.
  if (torch.cuda.is_available()):
    # Set the device to CUDA.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Print the device being used.
    fprint(f"Using device: {device}")
    # Calculate and print the GPU memory.
    gpuMemory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0
    # Print the GPU memory value.
    fprint(f"GPU Memory: {gpuMemory:.2f} GB")
  else:
    # Set the device to CPU if CUDA is not available.
    device = torch.device("cpu")
    # Print the device being used.
    fprint(f"Using device: {device}")
  # Import the argparse module for command-line interface.
  import argparse

  # Initialize the argument parser for command-line interface.
  parser = argparse.ArgumentParser(description="ViT Image Classification")
  # Add the command-line argument for the configuration file.
  parser.add_argument(
    "--config",
    type=str,
    default="config.pytorch.yaml",
    help="Path to the YAML or JSON configuration file"
  )
  # Parse the command-line arguments into a namespace object.
  args = parser.parse_args()
  # Load the configuration from the file.
  config = LoadConfig(args.config)
  # Extract base parameters for the experiment grid.
  baseOutputDir = str(Path(config.get("OutputDir", "Results")).parent)
  # Extract the batch size from the configuration.
  batchSize = config.get("BatchSize", 16)
  # Extract lists or single values for the experiment grid.
  modelNames = config.get("ModelName", "StandardViT")
  # Extract the list of seeds for multi-trial execution.
  seeds = config.get("Seeds", [42])
  # Extract the optimizers from the configuration.
  optimizers = config.get("Optimizer", "Adam")
  # Extract the loss functions from the configuration.
  lossFunctions = config.get("LossFunction", "CrossEntropy")
  # Extract the pretrained custom models flag from the configuration.
  usePretrainedCustomModels = config.get("UsePretrainedCustomModels", False)
  # Extract the pretrained model name from the configuration.
  pretrainedModelName = config.get("PretrainedModelName", "vit_base_patch16_224")
  # Extract the metric to judge the best model by from the configuration.
  judgeBy = config.get("JudgeBy", "val_accuracy")
  # Extract the save every parameter from the configuration.
  saveEvery = config.get("SaveEvery", None)
  # Extract the maximum gradient norm for gradient clipping from the configuration.
  maxGradNorm = config.get("MaxGradNorm", None)
  # Extract the resume from checkpoint flag from the configuration.
  resumeFromCheckpoint = config.get("ResumeFromCheckpoint", False)
  # Ensure the model names variable is a list for iteration.
  if (isinstance(modelNames, str)):
    # Convert a single string to a list.
    modelNames = [modelNames]
  # Ensure the optimizers variable is a list for iteration.
  if (isinstance(optimizers, str)):
    # Convert a single string to a list.
    optimizers = [optimizers]
  # Ensure the loss functions variable is a list for iteration.
  if (isinstance(lossFunctions, str)):
    # Convert a single string to a list.
    lossFunctions = [lossFunctions]
  # Ensure the seeds variable is a list for iteration.
  if (isinstance(seeds, int)):
    # Convert a single integer seed to a list.
    seeds = [seeds]
  # Loop through all combinations of models, optimizers, loss functions, and seeds.
  for modelName in modelNames:
    # Iterate over each optimizer in the configured list.
    for optimizerName in optimizers:
      # Iterate over each loss function in the configured list.
      for lossFunction in lossFunctions:
        # Iterate over each seed for the multi-trial execution.
        for seed in seeds:
          # Set the global seed for complete reproducibility.
          SeedEverything(seed)
          # Clear the CUDA cache to free up memory before starting a new experiment.
          if (torch.cuda.is_available()):
            # Empty the CUDA cache.
            torch.cuda.empty_cache()
          # Generate a clean model name for the folder path.
          folderModelName = modelName.replace("Transformer", "").replace("ViT", "")
          # Check if the folder model name is Standard.
          if (folderModelName == "Standard"):
            # Set the folder model name to ViT.
            folderModelName = "ViT"
          # Construct dynamic output directory name including the seed.
          experimentName = f"Exp-{folderModelName}-{optimizerName}-{batchSize}-{lossFunction}"
          # Construct the current output directory path with the seed subfolder.
          currentOutputDir = str(Path(baseOutputDir) / experimentName / f"Seed-{seed}")
          # Print a separator line for the new experiment.
          fprint(f"\n{'=' * 60}")
          # Print the starting experiment message.
          fprint(f"Starting Experiment: {experimentName} | Seed: {seed}")
          # Print the output directory message.
          fprint(f"Output Directory: {currentOutputDir}")
          # Print a separator line.
          fprint(f"{'=' * 60}\n")
          # Start the try block for experiment execution.
          try:
            # Execute the complete pipeline with parsed configuration parameters.
            RunCompletePipeline(
              dataDir=config.get("DataDir", "./data"),
              outputDir=currentOutputDir,
              modelName=modelName,
              numEpochs=config.get("NumEpochs", 50),
              batchSize=batchSize,
              imageSize=config.get("ImageSize", 224),
              learningRate=config.get("LearningRate", 1e-4),
              devicePref=config.get("Device", "auto"),
              patience=config.get("Patience", 10),
              optimizerName=optimizerName,
              useAmp=config.get("UseAmp", False),
              accumulationSteps=config.get("AccumulationSteps", 1),
              useEma=config.get("UseEma", False),
              useMixup=config.get("UseMixup", False),
              mixupAlpha=config.get("MixupAlpha", 0.2),
              cutmixAlpha=config.get("CutmixAlpha", 1.0),
              lossFunction=lossFunction,
              labelSmoothing=config.get("LabelSmoothing", 0.0),
              schedulerName=config.get("Scheduler", "None"),
              useAugmentation=config.get("UseAugmentation", False),
              useStainJitter=config.get("UseStainJitter", False),
              useBackgroundRemoval=config.get("UseBackgroundRemoval", False),
              useColorDeconvolution=config.get("UseColorDeconvolution", False),
              useTopoWassersteinLoss=config.get("UseTopoWassersteinLoss", True),
              epsilon=config.get("Epsilon", 0.1),
              lambdaTopo=config.get("LambdaTopo", 0.1),
              lambdaContrastive=config.get("LambdaContrastive", 0.1),
              usePretrainedCustomModels=usePretrainedCustomModels,
              pretrainedModelName=pretrainedModelName,
            )
            # Create a copy of the configuration for saving.
            configCopy = config.copy()
            # Update the output directory in the configuration copy.
            configCopy["OutputDir"] = currentOutputDir
            # Update the model name in the configuration copy.
            configCopy["ModelName"] = modelName
            # Update the optimizer in the configuration copy.
            configCopy["Optimizer"] = optimizerName
            # Update the loss function in the configuration copy.
            configCopy["LossFunction"] = lossFunction
            # Update the seed in the configuration copy.
            configCopy["Seed"] = seed
            # Open the configuration file for writing.
            with open(Path(currentOutputDir) / "ConfigUsed.yaml", "w") as f:
              # Dump the configuration copy to the YAML file.
              yaml.dump(configCopy, f)
          # Catch any exceptions during the experiment execution.
          except Exception as e:
            # Print the error message.
            fprint(f"ERROR in experiment {experimentName} with seed {seed}: {e}")
            # Import the traceback module for detailed error printing.
            import traceback

            # Print the detailed traceback.
            traceback.print_exc()
            # Print a message indicating continuation.
            fprint("Continuing to the next experiment...")
            # Continue to the next iteration.
            continue
  # Print a separator line for the final metrics.
  fprint("\n" + "=" * 60)
  # Print the completion message.
  fprint("All experiments completed.")
