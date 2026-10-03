import torch, traceback
from PIL import Image
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from torchvision import transforms
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Step1PyTorchPretrainedViTPipeline import BuildViTModel
from HMB.Utils import fprint
from HMB.PyTorchHelper import LoadModel
from HMB.Initializations import IMAGE_SUFFIXES
from HMB.ExplainabilityHelper import CAMExplainerPyTorch


def RunAdvancedXaiOnDataset(
  model: torch.nn.Module,
  datasetPath: str,
  splitName: str,
  outputDirectory: str,
  classNames: Dict[int, str],
  imageSize: int = 224,
  maxImagesPerClass=25,
  methods: List[str] = [
    "integratedgradients",
    "smoothgrad",
    "rise",
    "featureablation",
    "vitgradcam",
    "vitxgradcam",
    "viteigencam",
  ]
) -> None:
  '''
  Runs advanced XAI methods on a dataset split and generates a combined figure.
  '''

  # Convert the dataset path to a Path object.
  datasetPathObj = Path(datasetPath)
  # Construct the split path.
  splitPath = datasetPathObj / splitName

  # Check if the split path does not exist.
  if (not splitPath.exists()):
    # Raise a FileNotFoundError.
    raise FileNotFoundError(f"Split directory \"{splitPath}\" does not exist.")

  # Initialize the files' dictionary.
  files = {
    classKey: list((splitPath / className).rglob("*"))
    for classKey, className in classNames.items()
  }
  # Initialize the image files list.
  imageFiles = []

  # Iterate over the files' dictionary.
  for classKey, fileList in files.items():
    # Filter valid image files.
    validFiles = [
      fileObj for fileObj in fileList
      if fileObj.suffix.lower() in IMAGE_SUFFIXES
    ]
    # Limit the number of files per class.
    limitedFiles = validFiles[:maxImagesPerClass]
    # Extend the image files list.
    imageFiles.extend(limitedFiles)

  # Determine the device to use.
  deviceName = "cuda" if torch.cuda.is_available() else "cpu"
  # Print the device being used.
  fprint(f"Using device: {deviceName}")
  # Print the processing message.
  fprint(f"Processing {len(imageFiles)} images with {len(methods)} XAI methods...")

  # Convert the output directory to a Path object.
  outputDir = Path(outputDirectory)
  # Create the output directory if it does not exist.
  outputDir.mkdir(parents=True, exist_ok=True)

  # Define the image transformation pipeline.
  transform = transforms.Compose([
    transforms.Resize((imageSize, imageSize)),
    transforms.ToTensor()
  ])

  # Iterate over each image path.
  for imgPath in imageFiles:
    try:
      # Load the image and convert to RGB.
      img = Image.open(imgPath).convert("RGB")
      # Apply the transformation and add a batch dimension.
      imgTensor = transform(img).unsqueeze(0).to(deviceName)
      # Convert the processed tensor back to a PIL image to ensure exact pixel match with the model input.
      imgDisplay = transforms.ToPILImage()(imgTensor.squeeze(0).cpu())

      # Set the model to evaluation mode.
      model.eval()
      # Disable gradient calculation for prediction.
      with torch.no_grad():
        # Get model outputs.
        outputs = model(imgTensor)
        # Determine the predicted class.
        predClass = outputs.argmax(dim=1).item()
        # Compute the predicted probability.
        predProb = torch.softmax(outputs, dim=1)[0, predClass].item()

      # Initialize the true class name.
      trueClassName = "Unknown"
      # Iterate over class names mapping.
      for classKey, className in classNames.items():
        # Check if the class name is in the image path.
        if (className in str(imgPath)):
          # Update the true class name.
          trueClassName = className
          # Break the loop.
          break

      # Print the processing status.
      fprint(f"Processing: {imgPath.name} | True: {trueClassName} | Pred: {predClass} ({predProb:.2f})")

      # Initialize the results' dictionary.
      results = {}

      # Iterate through each CAM technique to apply explainability.
      for camMethod in methods:
        # Initialize the PyTorch CAM explainer with the current method.
        explainer = CAMExplainerPyTorch(
          torchModel=model,
          device=deviceName,
          camType=camMethod,
          imgSize=imageSize,
          debug=False,
        )
        # Compute the saliency map.
        mask = explainer.ComputeSaliency(imgTensor, predClass)
        # Format the method name using the explainer helper.
        formattedName = explainer.CamTypeToFolderName(camMethod)
        # Store the mask in the results' dictionary.
        results[formattedName] = mask
        fprint("Mask shape: ", mask.shape, " | Min: ", mask.min(), " | Max: ", mask.max())

      # Create a new figure for visualization.
      fig = plt.figure(figsize=(20, 5), constrained_layout=False)
      # Calculate the number of columns for the grid.
      numCols = len(results) + 1
      # Create a GridSpec for the figure.
      gs = gridspec.GridSpec(1, numCols, figure=fig, wspace=0.1)

      # Add the original image subplot.
      axOrig = fig.add_subplot(gs[0, 0])
      # Display the resized original image.
      axOrig.imshow(imgDisplay, extent=[0, imageSize, imageSize, 0])
      # Set the title for the original image.
      predClassName = classNames.get(predClass, "Unknown")
      axOrig.set_title(f"Original\nPred: {predClassName} ({predProb:.2f})", fontsize=12, fontweight="bold")
      # Turn off the axis for the original image.
      axOrig.axis("off")
      fprint("Original image shape: ", imgDisplay.size)

      # Iterate over the results to plot XAI masks.
      for idx, (methodName, mask) in enumerate(results.items()):
        # Squeeze the mask to remove singleton dimensions for matplotlib.
        mask = mask.squeeze()
        # Convert mask to numpy array if it is a tensor.
        if (isinstance(mask, torch.Tensor)):
          # Move the tensor to CPU and convert to numpy.
          mask = mask.cpu().numpy()
        # Add a subplot for the current method.
        ax = fig.add_subplot(gs[0, idx + 1])
        # Display the resized original image as background with explicit extent for perfect alignment.
        ax.imshow(imgDisplay, extent=[0, imageSize, imageSize, 0])
        # Overlay the mask on the original image with explicit extent to ensure perfect alignment.
        im = ax.imshow(
          mask, cmap="jet", alpha=0.5, interpolation="bicubic",
          extent=[0, imageSize, imageSize, 0], rasterized=True
        )
        # Set the title for the current method.
        ax.set_title(f"{methodName}", fontsize=12, fontweight="bold")
        # Create a divider for the axes to add a colorbar without shrinking the image.
        divider = make_axes_locatable(ax)
        # Append a new axes to the right of the main axes for the colorbar.
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # Add a colorbar for the mask in the new axes.
        cbar = plt.colorbar(im, cax=cax)
        # Set the y-ticks of the colorbar to be empty.
        cbar.ax.set_yticks([])
        # Turn off the axis for the current method.
        ax.axis("off")

      # Adjust the layout of the figure manually to avoid tight_layout warnings with colorbars.
      # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.1)

      # Construct the save path for the figure and save it as PNG.
      plt.savefig(outputDir / f"XAI_{imgPath.stem}.png", dpi=750, bbox_inches="tight")
      # Save the figure to the specified path as PDF.
      plt.savefig(outputDir / f"XAI_{imgPath.stem}.pdf", dpi=750, bbox_inches="tight")
      # Close the figure to free memory.
      plt.close(fig)

    except Exception as e:
      # Print the error message.
      fprint(f"Error processing {imgPath}: {e}")
      # Print the traceback.
      traceback.print_exc()
      # Continue to the next image.
      continue

  # Print the final completion message.
  fprint(f"Advanced XAI processing completed. Results saved to: {outputDir}")


if __name__ == "__main__":
  # Print the starting message.
  fprint("Starting Advanced ViT XAI Pipeline...")

  datasetPath = r"/path/to/your/dataset"  # Update this path to your dataset location.
  splitName = "test"  # Specify the dataset split to use (e.g., "train", "val", "test").
  modelCheckpointPath = r"/path/to/your/modelCheckpoint.pth"  # Update this path to your model checkpoint.
  outputDir = Path(r"/path/to/output/directory/xai")  # Update this path to your desired output directory.
  # Define the class names mapping.
  classNamesMapping: Dict[int, str] = {
    0: "Grade_1",
    1: "Grade_2",
    2: "Grade_3",
  }

  # Define the model name.
  modelName = "ConvNeXtV2"

  # Determine the device to use.
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # Define the effective image size.
  effectiveImageSize = 224
  # Define the maximum number of images per class to process.
  maxImagesPerClass = 5

  # Define the number of classes.
  numClasses = len(classNamesMapping)
  # Print the class names mapping.
  fprint(f"Class names mapping: {classNamesMapping}")

  # Print the model building message.
  fprint("Building model...")
  # Build the ViT model.
  model, secondaryModel = BuildViTModel(modelName, numClasses, device, effectiveImageSize)
  # Move the model to the specified device.
  model = model.to(device)
  # Load the model weights.
  model = LoadModel(model, modelCheckpointPath, device)
  # Set the model to evaluation mode.
  model.eval()

  # Check if the output directory does not exist.
  if (not outputDir.exists()):
    # Create the output directory.
    outputDir.mkdir(parents=True, exist_ok=True)
    # Print the creation message.
    fprint(f"Created output directory: {outputDir}")

  # Run the advanced XAI on the dataset.
  RunAdvancedXaiOnDataset(
    model=model,
    datasetPath=datasetPath,
    splitName="test",
    outputDirectory=str(outputDir),
    classNames=classNamesMapping,
    imageSize=effectiveImageSize,
    maxImagesPerClass=maxImagesPerClass,
    methods=[
      "integratedgradients",
      "smoothgrad",
      # "rise",
      "featureablation",
      "vitgradcam",
      "vitxgradcam",
      "viteigencam",
    ]
  )
