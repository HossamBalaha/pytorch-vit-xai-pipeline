import torch, pandas
from pathlib import Path
from HMB.PlotsHelper import EfficiencyPlotter
from HMB.Initializations import UpdateMatplotlibSettings
from HMB.PyTorchClassificationModelsZoo import BuildViTModel
from HMB.PyTorchModelMemoryProfiler import PyTorchModelMemoryProfiler
from HMB.Examples.ModelsInferenceEfficiencyProfiling import ProfileModelEfficiency


if (__name__ == "__main__"):
  # Update the matplotlib settings for consistent plotting.
  UpdateMatplotlibSettings()

  # Define the list of models to profile.
  modelNames = ["StandardViT", "EVA02", "ConvNeXtV2"]

  # Define the accuracy for each model to be used in the Pareto front.
  modelAccuracies = {
    "StandardViT": 0.9656,
    "EVA02"      : 0.9943,
    "ConvNeXtV2" : 0.9961
  }

  # Define the number of classes for the model output.
  numClasses = 8

  # Define the device to use for computation.
  device = "cuda" if (torch.cuda.is_available()) else "cpu"

  # Define the input image size for the models.
  imageSize = 224

  # Define the output directory for saving results and plots.
  outputDirectory = "./Experiments/EfficiencyProfiling"

  # Initialize a list to store the profiling results.
  profilingResults = []

  # Create a dummy input tensor for profiling the models.
  dummyInput = torch.randn(1, 3, imageSize, imageSize)

  # Iterate through each model name in the list.
  for modelName in modelNames:
    # Build the model using the predefined pipeline function.
    model, _ = BuildViTModel(modelName, numClasses, device, imageSize)

    # Profile the model efficiency using the HMB profiler.
    metrics = ProfileModelEfficiency(
      model,
      dummyInput,
      device,
      optimizerType="Adam",
      isTransformer=False,
      checkpointing=False,
      deviceFLOPSGFLOPS=None,
      datasetSize=None,
      trainingMultiplier=3.0,
      runMicroBenchmark=True
    )

    # Add the model name to the metrics dictionary.
    metrics["ModelName"] = modelName

    # Add the model accuracy to the metrics dictionary.
    metrics["Accuracy"] = modelAccuracies.get(modelName, 0.85)

    # Append the metrics dictionary to the results list.
    profilingResults.append(metrics)

    # Print the profiling status for the current model.
    print("Profiled " + modelName + ":")

    # Print the total parameter count.
    print("  Parameters: " + str(metrics["ParameterCount"]))

    # Print the average latency in milliseconds.
    print("  Latency: " + str(round(metrics["AverageLatencyMs"], 4)) + " ms")

    # Print the peak inference memory in megabytes.
    print("  Inference Memory: " + str(round(metrics["PeakMemoryMb"], 2)) + " MB")

    # Print the total training memory in megabytes.
    print("  Training Memory: " + str(round(metrics["TrainingMemoryMb"], 2)) + " MB")

    # Print the total GFLOPs.
    print("  GFLOPs: " + str(round(metrics["TotalGFLOPs"], 4)))

    # Calculate and print the inference throughput safely.
    throughputValue = metrics["InferenceSamplesPerSec"]
    throughputString = str(round(throughputValue, 2)) if (throughputValue) else "N/A"
    print("  Throughput: " + throughputString + " samples/sec")

  # Convert the results list to a pandas DataFrame for analysis.
  resultsDataFrame = pandas.DataFrame(profilingResults)

  # Create the efficiency plotter instance with the results.
  plotter = EfficiencyPlotter(resultsDataFrame, outputDirectory)

  # Generate the multi-metric Pareto front visualization.
  plotter.PlotParetoFrontMultiMetric()

  # Generate the parameter count comparison visualization.
  plotter.PlotParameterCount()

  # Generate the latency comparison visualization.
  plotter.PlatencyComparison()

  # Generate the stacked memory breakdown visualization.
  plotter.PlotMemoryBreakdownStacked()

  # Generate the GFLOPs comparison visualization.
  plotter.PlotGFLOPsComparison()

  # Generate the throughput comparison visualization.
  plotter.PlotThroughputComparison()

  # Generate the memory efficiency visualization.
  plotter.PlotMemoryEfficiency()

  # Generate the training versus inference memory visualization.
  plotter.PlotTrainingVsInferenceMemory()

  # Generate the metrics correlation heatmap visualization.
  plotter.PlotMetricsCorrelationHeatmap()

  # Generate the top layers FLOPs visualization.
  plotter.PlotTopLayersFlops()

  # Generate the efficiency summary dashboard visualization.
  plotter.PlotEfficiencySummaryDashboard()

  # Create the output path object for saving files.
  outputPath = Path(outputDirectory)

  # Create the output directory if it does not already exist.
  outputPath.mkdir(parents=True, exist_ok=True)

  # Iterate through each row in the results DataFrame.
  for _, row in resultsDataFrame.iterrows():
    # Create a profiler instance to handle JSON saving.
    profiler = PyTorchModelMemoryProfiler(
      model=None,
      inputShape=(3, imageSize, imageSize),
      batchSize=1,
      precision="FP32",
      device=device
    )

    # Define the CamelCase file name for the detailed profile.
    profileFileName = row["ModelName"] + "DetailedProfile.json"

    # Save the detailed memory profile to a JSON file.
    profiler.SaveProfileToJSON(row["MemoryProfile"], outputPath / profileFileName)

  # Print the completion message with the output path.
  print("Detailed profiles saved to " + str(outputPath))
