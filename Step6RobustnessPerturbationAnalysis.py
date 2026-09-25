import os, torch, numpy
from PIL import Image
from pathlib import Path
from torchvision import transforms
from HMB.Initializations import UpdateMatplotlibSettings
from HMB.PyTorchClassificationModelsZoo import BuildViTModel
from HMB.PyTorchHelper import LoadModel, EvaluateModelOnPerturbations
from Step1PyTorchPretrainedViTPipeline import PyTorchFolderBasedDataPipeline


# Define the function to create a prediction callable for the HMB evaluator.
def CreatePredictionCallable(model, device, imageSize):
  # Define the image transformation pipeline.
  transformPipeline = transforms.Compose([
    # Resize the image to the target size.
    transforms.Resize((imageSize, imageSize)),
    # Convert the image to a tensor.
    transforms.ToTensor()
  ])

  # Define the prediction function that accepts an image and optional extra arguments.
  def PredictFn(*args):
    # Extract the image argument from the first positional argument.
    imageInput = args[0]

    # Convert the input to a PIL Image regardless of the original type.
    if (isinstance(imageInput, torch.Tensor)):
      # Squeeze batch dimension if present and convert to PIL.
      tensorInput = imageInput.squeeze(0) if (imageInput.dim() == 4) else imageInput
      # Convert the tensor to a PIL Image.
      pilImage = transforms.ToPILImage()(tensorInput.cpu())
    elif (isinstance(imageInput, numpy.ndarray)):
      # Check if the numpy array is in CHW format and convert to HWC.
      if (imageInput.ndim == 3 and imageInput.shape[0] <= 4):
        # Transpose from CHW to HWC format.
        imageInput = numpy.transpose(imageInput, (1, 2, 0))
      # Check if the array is float and scale to uint8.
      if (imageInput.dtype == numpy.float32 or imageInput.dtype == numpy.float64):
        # Scale float values to the 0-255 range.
        imageInput = (imageInput * 255.0).clip(0, 255).astype(numpy.uint8)
      # Check if the image has an alpha channel and remove it.
      if (imageInput.ndim == 3 and imageInput.shape[2] == 4):
        # Drop the alpha channel to keep only RGB.
        imageInput = imageInput[:, :, :3]
      # Check if the image is grayscale and convert to RGB.
      if (imageInput.ndim == 2):
        # Stack grayscale into three channels.
        imageInput = numpy.stack([imageInput, imageInput, imageInput], axis=-1)
      # Convert the numpy array to a PIL Image.
      pilImage = Image.fromarray(imageInput)
    elif (isinstance(imageInput, Image.Image)):
      # Use the PIL Image directly.
      pilImage = imageInput
    else:
      # Raise an error for unsupported input types.
      raise TypeError("Unsupported input type: " + str(type(imageInput)))

    # Ensure the PIL Image is in RGB mode.
    if (pilImage.mode != "RGB"):
      # Convert to RGB mode.
      pilImage = pilImage.convert("RGB")

    # Apply the transformation pipeline and add a batch dimension.
    inputTensor = transformPipeline(pilImage).unsqueeze(0).to(device)

    # Disable gradient computation for inference.
    with torch.no_grad():
      # Perform the forward pass through the model.
      outputs = model(inputTensor)
      # Check if the output is a tuple and extract the logits.
      if (isinstance(outputs, tuple)):
        # Extract the first element as logits.
        outputs = outputs[0]
      # Apply softmax to get probabilities and move to CPU.
      probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

    # Return the 1D array of class probabilities for the first sample.
    return probabilities[0]

  # Return the prediction function.
  return PredictFn


# Define the function to run robustness analysis.
def RunRobustnessAnalysis(model, datasetDirectory, outputDirectory, imageSize, classNames, device):
  # Convert the output directory to a Path object.
  outputPath = Path(outputDirectory)
  # Create the output directory if it does not exist.
  outputPath.mkdir(parents=True, exist_ok=True)
  # Define the list of perturbations to evaluate.
  perturbationsList = ["gaussian", "brightness", "jpeg", "speckle", "contrast"]
  # Define the severity levels for each perturbation.
  severityLevels = [0.1, 0.2, 0.3, 0.4, 0.5]
  # Print the starting message.
  print("Starting robustness and perturbation analysis.")
  # Create the prediction callable for the HMB evaluator.
  predictFn = CreatePredictionCallable(model, device, imageSize)
  # Call the HMB helper to evaluate the model on perturbations.
  robustnessResults = EvaluateModelOnPerturbations(
    model=predictFn,
    run="RobustnessRun",
    datasetDir=datasetDirectory,
    storeDir=str(outputPath),
    perturbations=perturbationsList,
    levels=severityLevels,
    maxSamples=200,
    preprocessFn=None,
    subset="test",
    eps=1e-10,
    dpi=720,
  )
  # Print the completion message.
  print("Robustness analysis complete. Results saved to " + str(outputPath))
  # Return the results' dictionary.
  return robustnessResults


if (__name__ == "__main__"):
  # Update the matplotlib settings for consistent plotting.
  UpdateMatplotlibSettings()

  # Define the dataset directory.
  datasetDirectory = "./data"
  # Define the model checkpoint path.
  modelCheckpointPath = "./Experiments/BestModel.pt"
  # Define the output directory.
  outputDirectory = "./Experiments/Robustness"

  # Define the model name.
  modelName = "ConvNeXtV2"
  # Determine the device to use.
  device = "cuda" if (torch.cuda.is_available()) else "cpu"
  # Define the image size.
  imageSize = 224
  # Create the data pipeline to get class names.
  pipeline = PyTorchFolderBasedDataPipeline(dataDir=datasetDirectory, imageSize=imageSize)
  # Build the model.
  model, _ = BuildViTModel(modelName, len(pipeline.classNames), device, imageSize)
  # Load the model weights.
  model = LoadModel(model, modelCheckpointPath, device)
  # Set the model to evaluation mode.
  model.eval()
  # Move the model to the device.
  model = model.to(device)
  # Run the robustness analysis.
  RunRobustnessAnalysis(model, datasetDirectory, outputDirectory, imageSize, pipeline.classNames, device)
