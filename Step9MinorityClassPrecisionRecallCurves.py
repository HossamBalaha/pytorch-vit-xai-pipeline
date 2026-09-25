import os
import ast
import numpy
import pandas
from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from HMB.Utils import fprint
from HMB.PlotsHelper import SaveMatplotlibFigure
from HMB.Initializations import UpdateMatplotlibSettings
from HMB.PerformanceMetrics import CalculatePerformanceMetrics


def LoadPredictionsFromCsv(csvPath):
  # Read the CSV file into a pandas DataFrame.
  dataFrame = pandas.read_csv(csvPath)

  # Initialize lists to store actual labels and predicted probabilities.
  actualLabels = []

  # Initialize the list for predicted probabilities.
  predictedProbs = []

  # Iterate over each row in the DataFrame.
  for index, row in dataFrame.iterrows():
    # Append the actual label to the list.
    actualLabels.append(int(row["Actual"]))

    # Parse the string representation of the probability list.
    probList = ast.literal_eval(row["PredProb"])

    # Append the parsed probability list.
    predictedProbs.append(probList)

  # Convert the lists to numpy arrays.
  actualLabelsNp = numpy.array(actualLabels)

  # Convert probabilities to a 2D numpy array.
  predictedProbsNp = numpy.array(predictedProbs)

  # Return the actual labels and predicted probabilities.
  return actualLabelsNp, predictedProbsNp


def IdentifyMinorityClasses(actualLabels, minorityRatio=0.33):
  # Get unique labels and their counts.
  uniqueLabels, classCounts = numpy.unique(actualLabels, return_counts=True)

  # Calculate the threshold for minority classes.
  threshold = numpy.percentile(classCounts, minorityRatio * 100)

  # Identify classes with counts less than or equal to the threshold.
  minorityClasses = uniqueLabels[classCounts <= threshold]

  # Return the array of minority class indices.
  return minorityClasses


def PlotPrecisionRecallCurves(actualLabels, predictedProbs, classNames, outputDirectory, minorityClasses):
  # Convert the output directory to a Path object.
  outputPath = Path(outputDirectory)

  # Create the output directory if it does not exist.
  outputPath.mkdir(parents=True, exist_ok=True)

  # Create a matplotlib figure for the PRC plot.
  fig, ax = plt.subplots(figsize=(8, 6))

  # Get the number of classes.
  numClasses = predictedProbs.shape[1]

  # Generate a list of colors for the classes.
  colors = plt.cm.tab20(numpy.linspace(0, 1, max(numClasses, 20)))

  # Iterate over each class to compute and plot the PRC.
  for classIdx in range(numClasses):
    # Create a binary indicator for the current class.
    binaryLabels = (actualLabels == classIdx).astype(int)

    # Extract the predicted probabilities for the current class.
    classProbs = predictedProbs[:, classIdx]

    # Compute the precision-recall curve.
    precision, recall, _ = precision_recall_curve(binaryLabels, classProbs)

    # Compute the average precision score (AUPRC).
    auprc = average_precision_score(binaryLabels, classProbs)

    # Get the class name from the mapping.
    className = classNames.get(classIdx, "Class_" + str(classIdx))

    # Determine if the current class is a minority class.
    isMinority = classIdx in minorityClasses

    # Set line width based on minority status.
    lineWidth = 3.0 if (isMinority) else 1.0

    # Set alpha for non-minority classes to fade them slightly.
    lineAlpha = 1.0 if (isMinority) else 0.5

    # Set marker style for minority classes.
    markerStyle = "o" if (isMinority) else ""

    # Set markevery to avoid overcrowding markers.
    markEvery = 20 if (isMinority) else 0

    # Plot the precision-recall curve.
    ax.plot(
      recall,
      precision,
      lw=lineWidth,
      alpha=lineAlpha,
      color=colors[classIdx],
      marker=markerStyle,
      markersize=4,
      markevery=markEvery,
      label=className + " (AUPRC = " + str(round(auprc, 3)) + ")"
    )

  # Set the x-axis label.
  ax.set_xlabel("Recall", fontweight="bold")

  # Set the y-axis label.
  ax.set_ylabel("Precision", fontweight="bold")

  # Set the title of the plot.
  ax.set_title("Precision-Recall Curves (Minority Classes Highlighted)", fontsize=16, fontweight="bold")

  # Add a grid for better readability.
  ax.grid(True, linestyle="--", alpha=0.6)

  # Set the x-axis limits.
  ax.set_xlim([0.0, 1.05])

  # Set the y-axis limits.
  ax.set_ylim([0.0, 1.05])

  # Add the legend to the plot.
  ax.legend(loc="lower left", fontsize=9, title="Classes", title_fontsize=11)

  # Adjust the layout to prevent clipping.
  fig.tight_layout()

  # Define the base file name for saving.
  baseFileName = "PrecisionRecallCurves"

  # Save the figure as both high-resolution PNG and vector PDF using the HMB helper.
  SaveMatplotlibFigure(
    os.path.join(str(outputPath), baseFileName),
    fig=fig,
    dpi=300,
    show=False,
    exportPdf=True,
    exportPng=True
  )

  # Close the figure to free memory.
  plt.close(fig)

  # Print confirmation message for the saved plots.
  fprint("Precision-Recall Curves saved to " + str(outputPath))


if (__name__ == "__main__"):
  # Update the matplotlib settings for consistent plotting.
  UpdateMatplotlibSettings()

  # Define the path to the detailed predictions CSV file.
  csvPath = r"/path/to/detailed/predictions.csv"

  # Define the output directory for the PRC plots.
  outputDirectory = r"./Experiments/PRC_Analysis"

  # Define the class names mapping.
  classNamesMapping: Dict[int, str] = {
    0: "AMD",
    1: "CNV",
    2: "CSR",
    3: "DME",
    4: "DR",
    5: "DRUSEN",
    6: "MH",
    7: "NORMAL",
  }

  # Print the starting message.
  fprint("Starting Precision-Recall Curve analysis.")

  # Load the predictions from the CSV file.
  actualLabels, predictedProbs = LoadPredictionsFromCsv(csvPath)

  # Identify the minority classes in the test set.
  minorityClasses = IdentifyMinorityClasses(actualLabels, minorityRatio=0.33)

  # Print the identified minority classes.
  fprint("Identified minority classes: " + str([classNamesMapping.get(int(c), c) for c in minorityClasses]))

  # Generate hard predictions for overall metrics calculation.
  hardPredictions = numpy.argmax(predictedProbs, axis=1)

  # Calculate the confusion matrix for overall performance metrics.
  confMatrix = confusion_matrix(actualLabels, hardPredictions)

  # Calculate comprehensive performance metrics using the HMB helper.
  overallMetrics = CalculatePerformanceMetrics(confMatrix, addWeightedAverage=True)

  # Print the overall weighted F1 score to the console.
  fprint("Overall Weighted F1 Score: " + str(round(overallMetrics["Weighted F1"], 4)))

  # Print the overall Macro F1 score to the console.
  fprint("Overall Macro F1 Score: " + str(round(overallMetrics["Macro F1"], 4)))

  # Plot the Precision-Recall Curves with minority class highlighting.
  PlotPrecisionRecallCurves(actualLabels, predictedProbs, classNamesMapping, outputDirectory, minorityClasses)

  # Print the completion message.
  fprint("Precision-Recall Curve analysis complete.")
