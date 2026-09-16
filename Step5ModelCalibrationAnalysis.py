import torch
import numpy
from pathlib import Path
from HMB.PyTorchHelper import LoadModel
from HMB.PerformanceMetrics import ComputeECEPlotReliability


# Define the function to run calibration analysis on a dataset.
def RunCalibrationAnalysis(model, dataLoader, device, outputDirectory):
  # Initialize the list for all confidences.
  allConfidences = []
  # Initialize the list for all predictions.
  allPredictions = []
  # Initialize the list for all true labels.
  allLabels = []
  # Set the model to evaluation mode.
  model.eval()
  # Disable gradient computation.
  with torch.no_grad():
    # Iterate over the data loader.
    for inputs, labels, _ in dataLoader:
      # Move inputs to the specified device.
      inputs = inputs.to(device)
      # Move labels to the specified device.
      labels = labels.to(device)
      # Perform the forward pass.
      outputs = model(inputs)
      # Apply softmax to get probabilities.
      probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
      # Get the maximum probabilities.
      maxProbabilities = numpy.max(probabilities, axis=1)
      # Get the predicted classes.
      predictedClasses = numpy.argmax(probabilities, axis=1)
      # Extend the confidences list.
      allConfidences.extend(maxProbabilities)
      # Extend the predictions list.
      allPredictions.extend(predictedClasses)
      # Extend the true labels list.
      allLabels.extend(labels.cpu().numpy())
  # Convert the confidences list to a numpy array.
  allConfidences = numpy.array(allConfidences)
  # Convert the predictions list to a numpy array.
  allPredictions = numpy.array(allPredictions)
  # Convert the true labels list to a numpy array.
  allLabels = numpy.array(allLabels)
  # Convert the output directory to a Path object.
  outputPath = Path(outputDirectory)
  # Create the output directory if it does not exist.
  outputPath.mkdir(parents=True, exist_ok=True)
  # Compute the ECE and plot the reliability diagram using the HMB helper.
  ece, binAccs, binConfs, binCounts = ComputeECEPlotReliability(
    confidences=allConfidences,
    predictions=allPredictions,
    labels=allLabels,
    nBins=10,
    title="Reliability Diagram",
    save=True,
    fileName=str(outputPath / "ReliabilityDiagram.pdf"),
    display=False,
    dpi=300
  )
  # Print the ECE value.
  print(f"Expected Calibration Error: {ece:.4f}")
  # Print the completion message.
  print(f"Calibration analysis complete. Results saved to {outputPath}")


# Define the main execution block.
if (__name__ == "__main__"):
  # Import the pipeline module.
  from Step1PyTorchPretrainedViTPipeline import BuildViTModel, PyTorchFolderBasedDataPipeline

  # Define the dataset directory.
  datasetDirectory = "./data"
  # Define the model checkpoint path.
  modelCheckpointPath = "./Experiments/BestModel.pt"
  # Define the output directory.
  outputDirectory = "./Experiments/Calibration"
  # Define the model name.
  modelName = "EVA02"

  # Determine the device to use.
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # Define the image size.
  imageSize = 224
  # Create the data pipeline.
  pipeline = PyTorchFolderBasedDataPipeline(dataDir=datasetDirectory, imageSize=imageSize)
  # Build the model.
  model, _ = BuildViTModel(modelName, len(pipeline.classNames), device, imageSize)
  # Load the model weights.
  model = LoadModel(model, modelCheckpointPath, device)
  # Move the model to the device.
  model = model.to(device)
  # Run the calibration analysis.
  RunCalibrationAnalysis(model, pipeline.test, device, outputDirectory)
