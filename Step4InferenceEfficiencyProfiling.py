import torch
import pandas
from pathlib import Path
from HMB.PlotsHelper import EfficiencyPlotter
from HMB.PyTorchModelMemoryProfiler import PyTorchModelMemoryProfiler
from HMB.Initializations import UpdateMatplotlibSettings
from Step1PyTorchPretrainedViTPipeline import BuildViTModel


# Define the function to profile model efficiency using HMB profiler.
def ProfileModelEfficiency(model, inputTensor, device, numIterations=100):
  # Extract the input shape from the tensor excluding batch dimension.
  inputShape = tuple(inputTensor.shape[1:])
  # Create the HMB profiler instance with FP32 precision.
  profiler = PyTorchModelMemoryProfiler(
    model=model,
    inputShape=inputShape,
    batchSize=inputTensor.shape[0],
    precision="FP32",
    device=device
  )
  # Run comprehensive memory and performance profiling.
  memoryProfile = profiler.ProfileModelMemory(
    optimizerType="Adam",
    isTransformer=False,
    checkpointing=False,
    deviceFLOPSGFLOPS=None,
    datasetSize=None,
    trainingMultiplier=3.0,
    runMicroBenchmark=True
  )
  # Extract performance estimates from the profile.
  performanceEstimates = memoryProfile["PerformanceEstimates"]
  # Extract memory breakdown in MB.
  memoryBreakdownMB = memoryProfile["MemoryBreakdownMB"]
  # Extract FLOPs estimate.
  flopsEstimate = memoryProfile["FLOPsEstimate"]
  # Create efficiency metrics dictionary using HMB data.
  efficiencyMetrics = {
    "ParameterCount"        : memoryProfile["ModelInfo"]["TotalParameters"],
    "TrainableParameters"   : memoryProfile["ModelInfo"]["TrainableParameters"],
    "AverageLatencyMs"      : (performanceEstimates.get("TimePerInferenceSampleSec", 0) or 0) * 1000,
    "PeakMemoryMb"          : memoryBreakdownMB["TotalInferenceMemory"],
    "TrainingMemoryMb"      : memoryBreakdownMB["TotalTrainingMemory"],
    "TotalGFLOPs"           : flopsEstimate.get("TotalGFLOPs", 0),
    "InferenceSamplesPerSec": performanceEstimates.get("InferenceSamplesPerSecond", 0),
    "MemoryProfile"         : memoryProfile
  }
  # Return the comprehensive efficiency metrics dictionary.
  return efficiencyMetrics


# Define the main execution block.
if (__name__ == "__main__"):
  UpdateMatplotlibSettings()

  # Define the list of models to profile.
  modelNames = ["StandardViT", "EVA02", "ConvNeXtV2"]
  # Define the accuracy for each model (replace with actual evaluation).
  modelAccuracies = {
    "StandardViT": 0.9656,
    "EVA02"      : 0.9943,
    "ConvNeXtV2" : 0.9961
  }
  # Define the number of classes.
  numClasses = 8
  # Define the device to use.
  device = "cuda" if (torch.cuda.is_available()) else "cpu"
  # Define the image size.
  imageSize = 224
  # Define the output directory.
  outputDirectory = "./Experiments/EfficiencyProfiling"
  # Initialize a list to store the results.
  profilingResults = []
  # Create a dummy input tensor for profiling.
  dummyInput = torch.randn(1, 3, imageSize, imageSize)
  # Iterate through each model name.
  for modelName in modelNames:
    # Build the model.
    model, _ = BuildViTModel(modelName, numClasses, device, imageSize)
    # Profile the model efficiency using HMB profiler.
    metrics = ProfileModelEfficiency(model, dummyInput, device)
    # Add the model name to the metrics.
    metrics["ModelName"] = modelName
    # Add a dummy accuracy for the Pareto front (replace with actual evaluation).
    metrics["Accuracy"] = modelAccuracies.get(modelName, 0.85)
    # Append the metrics to the results list.
    profilingResults.append(metrics)
    # Print the profiling status.
    print("Profiled " + modelName + ":")
    print("  Parameters: " + str(metrics["ParameterCount"]))
    print("  Latency: " + str(round(metrics["AverageLatencyMs"], 4)) + " ms")
    print("  Inference Memory: " + str(round(metrics["PeakMemoryMb"], 2)) + " MB")
    print("  Training Memory: " + str(round(metrics["TrainingMemoryMb"], 2)) + " MB")
    print("  GFLOPs: " + str(round(metrics["TotalGFLOPs"], 4)))
    print("  Throughput: " + str(
      round(metrics["InferenceSamplesPerSec"], 2) if metrics["InferenceSamplesPerSec"] else "N/A") + " samples/sec")
  # Convert the results list to a pandas DataFrame.
  resultsDataFrame = pandas.DataFrame(profilingResults)

  # Create the efficiency plotter instance.
  plotter = EfficiencyPlotter(resultsDataFrame, outputDirectory)

  # Generate all visualizations.
  plotter.PlotParetoFrontMultiMetric()
  plotter.PlotParameterCount()
  plotter.PlatencyComparison()
  plotter.PlotMemoryBreakdownStacked()
  plotter.PlotGFLOPsComparison()
  plotter.PlotThroughputComparison()
  plotter.PlotMemoryEfficiency()
  plotter.PlotTrainingVsInferenceMemory()
  plotter.PlotMetricsCorrelationHeatmap()
  plotter.PlotTopLayersFlops()
  plotter.PlotEfficiencySummaryDashboard()

  # Save detailed profiles to JSON for each model.
  outputPath = Path(outputDirectory)
  outputPath.mkdir(parents=True, exist_ok=True)
  for _, row in resultsDataFrame.iterrows():
    profiler = PyTorchModelMemoryProfiler(
      model=None,
      inputShape=(3, imageSize, imageSize),
      batchSize=1,
      precision="FP32",
      device=device
    )
    profiler.SaveProfileToJSON(row["MemoryProfile"], outputPath / (row["ModelName"] + "_Profile.json"))
  print("Detailed profiles saved to " + str(outputPath))
