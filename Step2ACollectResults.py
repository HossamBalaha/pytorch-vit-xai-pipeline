# Import operating system and JSON support.
import os, json
# Import Path for filesystem paths.
from pathlib import Path
# Import required HMB utility functions.
from HMB.Utils import ConvertToCamelCase, AppendOrCreateNewCSV


# Collects validation and test results from experiment folders into CSV files.
def CollectResults(baseDir, outputCsvVal, outputCsvTest):
  # Initialize an empty list for validation results.
  valResults = []
  # Initialize an empty list for test results.
  testResults = []
  # Get all experiment directories in the base directory.
  experimentDirs = [d for d in Path(baseDir).iterdir() if (d.is_dir() and d.name.startswith("Exp-"))]
  # Iterate through each experiment directory.
  for expDir in experimentDirs:
    # Parse the configuration from the folder name.
    config = ParseFolderName(expDir.name)
    # Define the path for the validation metrics JSON file.
    valJsonPath = expDir / "Val" / "ValEvaluationMetrics.json"
    # Load the validation metrics from the JSON file.
    valMetrics = LoadMetrics(valJsonPath)
    # Check if validation metrics were successfully loaded.
    if (valMetrics):
      # Merge the configuration and validation metrics into a single row.
      valRow = {**config, **valMetrics}
      # Append the merged row to the validation results list.
      valResults.append(valRow)
    # Define the path for the test metrics JSON file.
    testJsonPath = expDir / "Test" / "TestEvaluationMetrics.json"
    # Load the test metrics from the JSON file.
    testMetrics = LoadMetrics(testJsonPath)
    # Check if test metrics were successfully loaded.
    if (testMetrics):
      # Merge the configuration and test metrics into a single row.
      testRow = {**config, **testMetrics}
      # Append the merged row to the test results list.
      testResults.append(testRow)
  # Write the collected validation results to a CSV file.
  WriteToCsv(valResults, outputCsvVal)
  # Write the collected test results to a CSV file.
  WriteToCsv(testResults, outputCsvTest)


# Extracts model, optimizer, batch size, and loss function from the folder name.
def ParseFolderName(folderName):
  # Split the folder name string by hyphens.
  parts = folderName.split("-")
  # Initialize an empty dictionary for the configuration.
  config = {}
  # Check if the folder name contains at least five parts.
  if (len(parts) >= 5):
    # Extract the model name from the second part.
    config["ModelName"] = parts[1]
    # Extract the optimizer name from the third part.
    config["OptimizerName"] = parts[2]
    # Extract the batch size from the fourth part.
    config["BatchSize"] = parts[3]
    # Extract the loss function from the fifth part.
    config["LossFunction"] = parts[4]
  # Handle unexpected folder name formats.
  else:
    # Store the raw folder name if the format is unexpected.
    config["ExperimentName"] = folderName
  # Return the parsed configuration dictionary.
  return config


# Loads and filters scalar metrics from a JSON evaluation file.
def LoadMetrics(jsonPath):
  # Check if the specified JSON file exists.
  if (not jsonPath.exists()):
    # Return an empty dictionary if the file is missing.
    return {}
  # Open the JSON file in read mode.
  with open(jsonPath, "r") as jsonFile:
    # Parse the JSON content into a Python dictionary.
    data = json.load(jsonFile)
  # Initialize an empty dictionary for the filtered metrics.
  metrics = {}
  # Iterate through all keys in the loaded JSON data.
  for key in data.keys():
    # Check if the value is a scalar type and not a list or dictionary.
    if (not isinstance(data[key], (list, dict))):
      # Convert the original key to CamelCase format.
      camelKey = ConvertToCamelCase(key)
      # Store the scalar metric value with the new CamelCase key.
      metrics[camelKey] = data[key]
  # Return the dictionary containing only scalar metrics.
  return metrics


# Writes a list of dictionaries to a CSV file by using the HMB utility.
def WriteToCsv(dataList, csvPath):
  # Check if the provided data list is empty.
  if (not dataList):
    # Print a warning message indicating no data to write.
    print("No data to write to " + str(csvPath) + ".")
    # Exit the function early since there is no data.
    return
  # Initialize an empty list to store the CSV column headers.
  headers = []
  # Iterate through each row dictionary in the data list.
  for row in dataList:
    # Iterate through each key in the current row dictionary.
    for key in row.keys():
      # Check if the key is not already present in the headers list.
      if (key not in headers):
        # Append the unique key to the headers list.
        headers.append(key)
  # Initialize an empty list to store row values aligned with the headers.
  rowList = []
  # Iterate through each row dictionary in the data list.
  for row in dataList:
    # Initialize an empty list for the current row values.
    rowValues = []
    # Iterate through each header in the collected header order.
    for header in headers:
      # Get the value for the current header or an empty string when missing.
      value = row.get(header, "")
      # Replace None values with empty strings for CSV compatibility.
      if (value is None):
        # Use an empty string for None values.
        value = ""
      # Append the prepared value to the current row values.
      rowValues.append(value)
    # Append the aligned row values to the row list.
    rowList.append(rowValues)
  # Remove an existing CSV file so the new results replace old results.
  if (os.path.exists(csvPath)):
    # Delete the existing CSV file.
    os.remove(csvPath)
  # Write the aligned rows through the HMB utility function.
  AppendOrCreateNewCSV(csvPath, rowList, header=headers)
  # Print a confirmation message indicating successful file creation.
  print("Results saved to " + str(csvPath) + ".")


# Execute the script when run directly.
if (__name__ == "__main__"):
  # Define the base directory where the experiment folders are located.
  baseDir = r"/path/to/your/base/directory"  # Update this path to your actual base directory.
  # Define the base directory where the experiment folders are located.
  baseDirectory = rf"{baseDir}/ExpResults"  # Update this path to your actual base directory.
  # Define the output file name for the validation results.
  valOutputFile = rf"{baseDir}/ValResults.csv"
  # Define the output file name for the test results.
  testOutputFile = rf"{baseDir}/TestResults.csv"
  # Execute the main function to collect and save the results.
  CollectResults(baseDirectory, valOutputFile, testOutputFile)
