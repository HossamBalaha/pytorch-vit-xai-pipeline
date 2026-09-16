# Import the json module for reading evaluation metrics.
import json
# Import the pandas module for data manipulation.
import pandas
# Import the Path class for object-oriented filesystem paths.
from pathlib import Path


# Define the function to aggregate results from multiple trials.
def AggregateMultiTrialResults(baseDirectory):
  # Initialize a list to store the collected metrics.
  collectedMetrics = []
  # Convert the base directory to a Path object.
  basePath = Path(baseDirectory)
  # Print the base path being searched for debugging.
  print(f"Searching for experiments in: {basePath}")
  # Check if the base path actually exists.
  if (not basePath.exists()):
    # Print an error if the directory is missing.
    print("ERROR: The specified base directory does not exist!")
    # Return None to indicate failure.
    return None
  # Iterate over all experiment directories in the base path.
  for expDir in basePath.iterdir():
    # Check if the current item is a directory.
    if (expDir.is_dir()):
      # Check if the directory name starts with "Exp-" (case-sensitive).
      if (expDir.name.startswith("Exp-")):
        # Print the found experiment directory.
        print(f"  Found Experiment Folder: {expDir.name}")
        # Iterate over ALL subdirectories in the experiment directory.
        for runDir in expDir.iterdir():
          # Check if the current item is a directory.
          if (runDir.is_dir()):
            # Define the possible paths to check based on the folder structure.
            possiblePaths = []
            # If the folder is named "Test", the JSON is directly inside.
            if (runDir.name == "Test"):
              # Append the Test evaluation metrics path.
              possiblePaths.append(runDir / "TestEvaluationMetrics.json")
            # If the folder is named "Val", the JSON is directly inside.
            elif (runDir.name == "Val"):
              # Append the Val evaluation metrics path.
              possiblePaths.append(runDir / "ValEvaluationMetrics.json")
            # If the folder is named "Train", the JSON is directly inside.
            elif (runDir.name == "Train"):
              # Append the Train evaluation metrics path.
              possiblePaths.append(runDir / "TrainEvaluationMetrics.json")
            # Execute this block if the folder is not Test, Val, or Train.
            else:
              # Otherwise, assume it is a Seed/Trial folder and look inside its subfolders.
              # Append the Test evaluation metrics path for the seed folder.
              possiblePaths.append(runDir / "Test" / "TestEvaluationMetrics.json")
              # Append the Val evaluation metrics path for the seed folder.
              possiblePaths.append(runDir / "Val" / "ValEvaluationMetrics.json")
              # Append the Train evaluation metrics path for the seed folder.
              possiblePaths.append(runDir / "Train" / "TrainEvaluationMetrics.json")

            # Iterate through possible paths and load the first one found.
            for metricsPath in possiblePaths:
              # Print the exact path being checked.
              print(f"    Checking run folder \"{runDir.name}\" for metrics at: {metricsPath}")
              # Check if the metrics file exists.
              if (metricsPath.exists()):
                # Open the metrics file for reading.
                with open(metricsPath, "r") as metricsFile:
                  # Load the JSON data into a dictionary.
                  metricsData = json.load(metricsFile)
                  # Add the experiment name to the metrics data.
                  metricsData["ExperimentName"] = expDir.name
                  # Add the run folder name to the metrics data.
                  metricsData["RunName"] = runDir.name
                  # Add the split name to the metrics data.
                  metricsData["SplitName"] = runDir.name
                  # Append the metrics data to the collected list.
                  collectedMetrics.append(metricsData)
                  # Print a success message.
                  print("      -> Successfully loaded metrics.")
                  # Break the loop since we found the file.
                  break
              # Execute this block if the metrics file does not exist.
              else:
                # Execute this block if the specific JSON file is missing.
                # Print a warning if the specific JSON file is missing.
                print("      -> WARNING: File not found.")
      # Execute this block if the directory name does not start with "Exp-".
      else:
        # Execute this block for unexpected experiment folder names.
        # Print a warning for unexpected experiment folder names.
        print(f"  Ignoring non-Exp folder: {expDir.name}")

  # Convert the collected metrics list to a pandas DataFrame.
  resultsDataFrame = pandas.DataFrame(collectedMetrics)
  # Check if the DataFrame is empty.
  if (resultsDataFrame.empty):
    # Print a message indicating no results were found.
    print("\nNo results found to aggregate. Please check the paths above.")
    # Return None to indicate failure.
    return None

  # Define the list of target metrics to aggregate.
  targetMetrics = [
    "Weighted Accuracy",
    "Weighted Recall",
    "Weighted Specificity",
    "Weighted Precision",
    "Weighted F1",
    "Weighted BAC",
    "AUC_ROC_Macro"
  ]

  # Initialize a dictionary to store the aggregated statistics.
  aggregatedStats = {}
  # Iterate over each unique experiment name in the DataFrame.
  for expName in resultsDataFrame["ExperimentName"].unique():
    # Filter the DataFrame for the current experiment.
    expData = resultsDataFrame[resultsDataFrame["ExperimentName"] == expName]
    # Initialize a dictionary for the current experiment stats.
    aggregatedStats[expName] = {}
    # Iterate over each target metric.
    for metric in targetMetrics:
      # Check if the metric exists in the experiment data.
      if (metric in expData.columns):
        # Convert the metric column to numeric, coercing errors to NaN.
        numericMetricData = pandas.to_numeric(expData[metric], errors="coerce")
        # Calculate the mean of the metric.
        metricMean = numericMetricData.mean()
        # Calculate the standard deviation of the metric.
        metricStd = numericMetricData.std()
        # Handle NaN standard deviation if only one run is present.
        if (pandas.isna(metricStd)):
          # Set standard deviation to zero.
          metricStd = 0.0
        # Store the mean and standard deviation in the stats dictionary.
        aggregatedStats[expName][metric] = f"{metricMean:.4f} ± {metricStd:.4f}"

  # Convert the aggregated statistics dictionary to a DataFrame.
  summaryDataFrame = pandas.DataFrame.from_dict(aggregatedStats, orient="index")
  # Define the path for the summary CSV file.
  summaryFilePath = basePath / "AggregatedResultsSummary.csv"
  # Save the summary DataFrame to a CSV file.
  summaryDataFrame.to_csv(summaryFilePath)
  # Print a completion message for the summary file.
  print(f"\nAggregation complete. Summary saved to {summaryFilePath}")

  # Define the columns to keep for the raw results file.
  rawColumns = ["ExperimentName", "RunName", "SplitName"] + targetMetrics
  # Filter the results DataFrame to keep only the relevant columns.
  rawResultsDataFrame = resultsDataFrame[rawColumns]
  # Define the path for the raw results CSV file.
  rawResultsFilePath = basePath / "RawResults.csv"
  # Save the raw results DataFrame to a CSV file without averaging.
  rawResultsDataFrame.to_csv(rawResultsFilePath, index=False)
  # Print a completion message for the raw results file.
  print(f"Raw results saved to {rawResultsFilePath}")

  # Return the summary DataFrame.
  return summaryDataFrame


# Define the main execution block.
if (__name__ == "__main__"):
  # Define the base directory containing the experiments.
  # Update this path to your actual experiments' directory.
  baseExperimentDir = r"/path/to/your/experiments"
  # Call the aggregation function.
  AggregateMultiTrialResults(baseExperimentDir)
