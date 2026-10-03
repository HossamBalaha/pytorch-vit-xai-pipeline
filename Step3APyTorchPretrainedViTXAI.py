import torch
from pathlib import Path
from HMB.Utils import fprint
from HMB.Initializations import IMAGE_SUFFIXES
from HMB.ExplainabilityHelper import CAMExplainerPyTorch
from Step1PyTorchPretrainedViTPipeline import BuildViTModel


# Define the function to run PyTorch CAM explainability on a dataset split using a timm model.
def RunTimmCamExplainabilityOnDataset(
  timmModel,
  datasetPath,
  splitName,
  outputDirectory,
  classNames,
  imageSize=224,
  maxImagesPerClass=25,
):
  # Define the list of available CAM techniques from the HMB package.
  availableCamMethods = [
    "gradcam",
    "gradcampp",
    "xgradcam",
    "eigencam",
    "layercam",
    "scorecam",
    "ablationcam",
    "saliency",
    "smoothgrad",
    "integratedgradients",
    "occlusion",
    "gradxinput",
    "smoothgradcampp",
    "rise",
    "featureablation",
    "vitgradcam",
    "vitxgradcam",
    "viteigencam",
  ]

  # Convert the dataset path to a Path object.
  datasetPathObj = Path(datasetPath)

  splitPath = datasetPathObj / splitName
  if (not splitPath.exists()):
    # Raise an error if the specified split directory does not exist.
    raise FileNotFoundError(f"Split directory \"{splitPath}\" does not exist.")

  files = {
    classIdx: list((splitPath / className).rglob("*"))
    for classIdx, className in classNames.items()
  }

  # Create a list to hold the selected image files for explainability.
  imageFiles = []
  for classIdx, fileList in files.items():
    # Filter the file list to include only valid image files based on their extensions.
    validFiles = [f for f in fileList if f.suffix.lower() in IMAGE_SUFFIXES]
    # Limit the number of images per class to the specified maximum.
    limitedFiles = validFiles[:maxImagesPerClass]
    # Extend the main image files list with the selected files for this class.
    imageFiles.extend(limitedFiles)

  # Determine the computation device based on CUDA availability.
  if (torch.cuda.is_available()):
    # Set the device to CUDA.
    deviceName = "cuda"
  else:
    # Set the device to CPU.
    deviceName = "cpu"

  # Iterate through each CAM technique to apply explainability.
  for camMethod in availableCamMethods:
    # Initialize the PyTorch CAM explainer with the current method.
    explainer = CAMExplainerPyTorch(
      torchModel=timmModel,
      device=deviceName,
      camType=camMethod,
      imgSize=imageSize,
      outputBase=outputDirectory,
      alpha=0.45,
      debug=False,
      figsize=(16, 14),
    )

    # Process the directory of images using the initialized explainer.
    results = explainer.ProcessDirectory(
      imageFiles=imageFiles,
      classNames=classNames,
    )

    # Print the result summary for the current CAM technique.
    fprint(f"Completed {camMethod} for timm model on {len(results)} images.")


# Define the main execution block.
if (__name__ == "__main__"):
  datasetPath = r"/path/to/your/dataset"  # Update this path to your dataset location.
  splitName = "test"  # Specify the dataset split to use (e.g., "train", "val", "test").
  modelCheckpointPath = r"/path/to/your/modelCheckpoint.pth"  # Update this path to your model checkpoint.
  outputDirectory = Path(r"/path/to/output/directory/xai")  # Update this path to your desired output directory.

  modelName = "SwinTransformerV2"
  device = "cuda" if (torch.cuda.is_available()) else "cpu"
  effectiveImageSize = 256
  maxImagesPerClass = 25

  # Define the class names mapping for the annotations using integer keys and CamelCase values.
  classNamesMapping = {
    0: "Healthy",
    1: "Tumor",
  }
  numClasses = len(classNamesMapping)
  fprint(f"Class names mapping: {classNamesMapping}.")

  # Build the ViT model with the specified image size.
  # secondaryModel is used for models like CLIP that require a separate visual encoder.
  model, secondaryModel = BuildViTModel(modelName, numClasses, device, effectiveImageSize)
  # Move the model to the specified device.
  model = model.to(device)
  # Load the best model weights from the checkpoint.
  model.load_state_dict(torch.load(modelCheckpointPath))

  # Set the model to evaluation mode.
  model.eval()

  if (not outputDirectory.exists()):
    # Create the output directory if it does not exist.
    outputDirectory.mkdir(parents=True, exist_ok=True)
    fprint(f"Created output directory: {outputDirectory}.")

  # Run the PyTorch CAM explainability pipeline on the dataset using the timm model.
  fprint("Running PyTorch CAM explainability pipeline...")
  RunTimmCamExplainabilityOnDataset(
    timmModel=model,
    datasetPath=datasetPath,
    splitName=splitName,
    outputDirectory=outputDirectory,
    classNames=classNamesMapping,
    imageSize=effectiveImageSize,
    maxImagesPerClass=maxImagesPerClass,
  )
