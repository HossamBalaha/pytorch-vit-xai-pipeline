import os, json, numpy, pandas
from pathlib import Path
from scipy import stats
from HMB.Utils import fprint
from HMB.StatisticalAnalysisHelper import PlotMetrics
from HMB.Initializations import UpdateMatplotlibSettings


# Define the function to aggregate metrics from multiple seeds.
def AggregateMetricsFromSeeds(experimentsDirectory, metricKey):
  # Initialize the dictionary to store aggregated metrics.
  aggregatedMetrics = {}
  # Convert the experiments directory to a Path object.
  experimentsPath = Path(experimentsDirectory)
  # Iterate through each seed directory.
  for seedDir in experimentsPath.iterdir():
    # Check if the current path is a directory.
    if (not seedDir.is_dir()):
      # Continue to the next directory.
      continue
    # Define the path to the evaluation metrics JSON file.
    metricsFilePath = seedDir / "Test" / "TestEvaluationMetrics.json"
    # Check if the metrics file exists.
    if (not metricsFilePath.exists()):
      # Continue to the next directory.
      continue
    # Open the metrics file for reading.
    with open(metricsFilePath, "r") as jsonFile:
      # Load the JSON data.
      metricsData = json.load(jsonFile)
    # Extract the seed name from the directory name.
    seedName = seedDir.name
    # Initialize the list for the seed if it does not exist.
    if (seedName not in aggregatedMetrics):
      # Create an empty list for the seed.
      aggregatedMetrics[seedName] = []
    # Append the metric value to the seed list.
    aggregatedMetrics[seedName].append(metricsData.get(metricKey, 0.0))
  # Return the aggregated metrics dictionary.
  return aggregatedMetrics


# Define the function to perform statistical analysis and plotting.
def PerformStatisticalAnalysis(aggregatedMetrics, outputDirectory, metricName):
  # Convert the output directory to a Path object.
  outputPath = Path(outputDirectory)
  # Create the output directory if it does not exist.
  outputPath.mkdir(parents=True, exist_ok=True)
  # Convert the aggregated metrics to a pandas DataFrame.
  metricsDataFrame = pandas.DataFrame(dict([(k, pandas.Series(v)) for k, v in aggregatedMetrics.items()]))
  # Print the starting message.
  fprint("Performing statistical analysis for " + metricName + ".")
  # Call the HMB PlotMetrics helper to generate statistical plots.
  PlotMetrics(
    data=metricsDataFrame,
    metricNames=list(aggregatedMetrics.keys()),
    outputDir=str(outputPath),
    plotTypes=["boxplot", "violin", "raincloud"],
    title=metricName + " Distribution Across Seeds",
    xLabel="Seed",
    yLabel=metricName,
    xLabelAlignment="center",
  )
  # Perform pairwise t-tests between the first seed and all other seeds.
  seedNames = list(aggregatedMetrics.keys())
  # Check if there are at least two seeds.
  if (len(seedNames) >= 2):
    # Iterate through the remaining seeds.
    for i in range(1, len(seedNames)):
      # Perform the independent t-test.
      tStat, pValue = stats.ttest_ind(aggregatedMetrics[seedNames[0]], aggregatedMetrics[seedNames[i]])
      # Calculate Cohen's d effect size.
      meanDiff = numpy.mean(aggregatedMetrics[seedNames[0]]) - numpy.mean(aggregatedMetrics[seedNames[i]])
      # Calculate the pooled standard deviation.
      pooledStd = numpy.sqrt(
        (numpy.std(aggregatedMetrics[seedNames[0]]) ** 2 + numpy.std(aggregatedMetrics[seedNames[i]]) ** 2) / 2)
      # Check if pooled standard deviation is greater than zero.
      if (pooledStd > 0):
        # Calculate Cohen's d.
        cohensD = meanDiff / pooledStd
      else:
        # Set Cohen's d to zero.
        cohensD = 0.0
      # Print the statistical results.
      fprint(
        "Comparison: " + seedNames[0] + " vs " + seedNames[i] +
        " | p-value: " + str(pValue) + " | Cohen's d: " + str(cohensD)
      )
  # Print the completion message.
  fprint("Statistical analysis complete. Results saved to " + str(outputPath))


# Define the main execution block.
if (__name__ == "__main__"):
  # Update the matplotlib settings for consistent plotting.
  UpdateMatplotlibSettings()
  # Define the base experiments directory containing seed folders.
  experimentsDirectory = "./Experiments/Exp-EVA02-AdamW-16-CrossEntropy"
  # Define the output directory for statistical analysis.
  outputDirectory = "./Experiments/StatisticalAnalysis"

  # Define the metric key to analyze.
  metricKey = "WeightedAccuracy"
  # Aggregate the metrics from all seed directories.
  aggregatedMetrics = AggregateMetricsFromSeeds(experimentsDirectory, metricKey)
  # Perform the statistical analysis and generate plots.
  PerformStatisticalAnalysis(aggregatedMetrics, outputDirectory, metricKey)
