import os
import ast
import numpy
import pandas
from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from HMB.Utils import fprint


# Define the function to load predictions from the detailed CSV file.
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


# Define the function to identify minority classes based on class support.
def IdentifyMinorityClasses(actualLabels, minorityRatio=0.33):
  # Get unique labels and their counts.
  uniqueLabels, classCounts = numpy.unique(actualLabels, return_counts=True)
  # Calculate the threshold for minority classes.
  threshold = numpy.percentile(classCounts, minorityRatio * 100)
  # Identify classes with counts less than or equal to the threshold.
  minorityClasses = uniqueLabels[classCounts <= threshold]
  # Return the array of minority class indices.
  return minorityClasses


# Define the function to plot Precision-Recall Curves.
def PlotPrecisionRecallCurves(actualLabels, predictedProbs, classNames, outputDirectory, minorityClasses):
  # Convert the output directory to a Path object.
  outputPath = Path(outputDirectory)
  # Create the output directory if it does not exist.
  outputPath.mkdir(parents=True, exist_ok=True)
  # Create a matplotlib figure for the PRC plot.
  fig, ax = plt.subplots(figsize=(10, 8))
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
  ax.set_xlabel("Recall", fontsize=14, fontweight="bold")
  # Set the y-axis label.
  ax.set_ylabel("Precision", fontsize=14, fontweight="bold")
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
  # Define the save path for the PNG file.
  pngPath = outputPath / "PrecisionRecallCurves.png"
  # Define the save path for the PDF file.
  pdfPath = outputPath / "PrecisionRecallCurves.pdf"
  # Save the figure as a high-resolution PNG.
  plt.savefig(str(pngPath), dpi=300, bbox_inches="tight")
  # Save the figure as a vector PDF.
  plt.savefig(str(pdfPath), dpi=300, bbox_inches="tight")
  # Close the figure to free memory.
  plt.close(fig)
  # Print confirmation message for the saved plots.
  fprint("Precision-Recall Curves saved to " + str(outputPath))


# Define the main execution block.
if (__name__ == "__main__"):
  # Define the path to the detailed predictions CSV file.
  # Update this path to your actual CSV file.
  csvPath = r"/path/to/detailed/predictions.csv"
  # Define the output directory for the PRC plots.
  outputDirectory = r"./Experiments/PRC_Analysis"
  # Define the class names mapping.
  classNamesMapping: Dict[int, str] = {
    0: "Grade_1",
    1: "Grade_2",
    2: "Grade_3",
  }
  # Print the starting message.
  fprint("Starting Precision-Recall Curve analysis.")
  # Load the predictions from the CSV file.
  actualLabels, predictedProbs = LoadPredictionsFromCsv(csvPath)
  # Identify the minority classes in the test set.
  minorityClasses = IdentifyMinorityClasses(actualLabels, minorityRatio=0.33)
  # Print the identified minority classes.
  fprint("Identified minority classes: " + str([classNamesMapping.get(c, c) for c in minorityClasses]))
  # Plot the Precision-Recall Curves.
  PlotPrecisionRecallCurves(actualLabels, predictedProbs, classNamesMapping, outputDirectory, minorityClasses)
  # Print the completion message.
  fprint("Precision-Recall Curve analysis complete.")
