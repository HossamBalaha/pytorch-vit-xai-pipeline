import os
import torch
import numpy
import pickle
import pandas
from tqdm import tqdm
from pathlib import Path
from Step1PyTorchPretrainedViTPipeline import BuildViTModel, PyTorchFolderBasedDataPipeline
from HMB.Utils import fprint
from HMB.PyTorchHelper import LoadModel
from HMB.ExplainabilityHelper import TSNEFeaturesExplainability, UMAPFeaturesExplainability


# Define the function to extract and save embeddings using a custom loop.
def ExtractAndSaveEmbeddings(model, dataPipeline, outputDirectory, device):
  # Convert the output directory to a Path object.
  outputPath = Path(outputDirectory)
  # Create the output directory if it does not exist.
  outputPath.mkdir(parents=True, exist_ok=True)
  # Print the starting message.
  fprint("Starting custom embedding extraction.")
  # Initialize lists to store embeddings, labels, predictions, and filenames.
  allEmbeddings = []
  # Initialize the labels list.
  allLabels = []
  # Initialize the predictions list.
  allPreds = []
  # Initialize the filenames list.
  allFilenames = []
  # Set the model to evaluation mode.
  model.eval()
  # Use the test dataloader from the pipeline.
  dataLoader = dataPipeline.test
  # Check if the test dataloader is None and fallback to validation or training.
  if (dataLoader is None):
    # Fallback to validation loader if available.
    dataLoader = dataPipeline.val if (dataPipeline.val is not None) else dataPipeline.train
  # Disable gradient computation for inference.
  with torch.no_grad():
    # Iterate over the dataloader with a progress bar.
    for inputs, labels, filenames in tqdm(dataLoader, desc="Extracting Embeddings"):
      # Move inputs to the specified device.
      inputs = inputs.to(device)
      # Extract features using the timm model's forward_features method.
      features = model.forward_features(inputs)
      # Check if the features have a sequence dimension (e.g., from Vision Transformers).
      if (features.dim() == 3):
        # Extract the CLS token which is typically at index 0 of the sequence dimension.
        embeddings = features[:, 0, :]
      # Check if the features have spatial dimensions for CNNs.
      elif (features.dim() == 4):
        # Average the features over the spatial dimensions for Convolutional Neural Networks.
        embeddings = features.mean(dim=(2, 3))
      # Handle other architectures.
      else:
        # Use the features as they are for other architectures.
        embeddings = features
      # Get model predictions for misclassification analysis in the latent space.
      outputs = model(inputs)
      # Handle tuple outputs from wrapped models.
      if (isinstance(outputs, tuple)):
        # Extract the logits.
        outputs = outputs[0]
      # Get the predicted class indices.
      preds = torch.argmax(outputs, dim=1).cpu().numpy()
      # Move embeddings to CPU and convert to a numpy array.
      embeddingsNp = embeddings.cpu().numpy()
      # Extend the accumulation lists.
      allEmbeddings.append(embeddingsNp)
      # Extend the labels list.
      allLabels.extend(labels.numpy())
      # Extend the predictions list.
      allPreds.extend(preds)
      # Extend the filenames list.
      allFilenames.extend(filenames)
  # Concatenate all embeddings into a single numpy array.
  allEmbeddings = numpy.concatenate(allEmbeddings, axis=0)
  # Convert labels and predictions to numpy arrays.
  allLabels = numpy.array(allLabels)
  # Convert predictions to a numpy array.
  allPreds = numpy.array(allPreds)
  # Create a dictionary to store the extracted data.
  embeddingsDict = {
    # Store the embeddings array.
    "Embeddings": allEmbeddings,
    # Store the labels array.
    "Labels": allLabels,
    # Store the predictions array.
    "Predictions": allPreds,
    # Store the filenames list.
    "Filenames": allFilenames,
    # Store the class names list.
    "ClassNames": dataPipeline.classNames
  }
  # Define the path for the pickle file.
  picklePath = outputPath / "Embeddings.pkl"
  # Open the file in write-binary mode and save the dictionary.
  with open(picklePath, "wb") as f:
    # Dump the dictionary to the pickle file.
    pickle.dump(embeddingsDict, f)
  # Print confirmation message for the pickle file.
  fprint("Embeddings saved to " + str(picklePath))
  # Create a dictionary for the metadata CSV file.
  csvData = {
    # Store the filenames.
    "Filename": allFilenames,
    # Store the labels.
    "Label": allLabels,
    # Store the predictions.
    "Prediction": allPreds,
    # Store the class names mapped from labels.
    "ClassName": [dataPipeline.classNames[l] for l in allLabels]
  }
  # Convert the dictionary to a pandas DataFrame.
  csvDataFrame = pandas.DataFrame(csvData)
  # Define the path for the CSV file.
  csvPath = outputPath / "Embeddings_Metadata.csv"
  # Save the DataFrame to a CSV file.
  csvDataFrame.to_csv(csvPath, index=False)
  # Print confirmation message for the CSV file.
  fprint("Metadata saved to " + str(csvPath))
  # Print the message indicating the start of dimensionality reduction.
  fprint("Starting dimensionality reduction and latent space analysis using HMB helpers.")
  # Define the maximum number of samples to plot to avoid memory issues.
  maxSamplesForPlot = 5000
  # Check if the total embeddings exceed the maximum limit.
  if (len(allEmbeddings) > maxSamplesForPlot):
    # Print a warning about sampling.
    fprint("Too many samples for t-SNE/UMAP. Randomly sampling " + str(maxSamplesForPlot) + " instances.")
    # Generate random indices for sampling.
    sampleIndices = numpy.random.choice(len(allEmbeddings), maxSamplesForPlot, replace=False)
    # Subset the embeddings for plotting.
    plotEmbeddings = allEmbeddings[sampleIndices]
    # Subset the labels for plotting.
    plotLabels = allLabels[sampleIndices]
    # Subset the predictions for plotting.
    plotPreds = allPreds[sampleIndices]
  # Handle the case where all samples can be plotted.
  else:
    # Use all embeddings for plotting.
    plotEmbeddings = allEmbeddings
    # Use all labels for plotting.
    plotLabels = allLabels
    # Use all predictions for plotting.
    plotPreds = allPreds
  # Compute t-SNE embeddings and generate publication-ready visualizations using HMB.
  fprint("Computing t-SNE and generating enhanced visualizations...")
  # Call the HMB t-SNE helper to generate interactive and static plots.
  tsneMetrics = TSNEFeaturesExplainability(
    # Pass the subset of embeddings.
    featsSub=plotEmbeddings,
    # Pass None since we provide custom class names.
    labelEncoder=None,
    # Pass the number of samples.
    nSamples=len(plotEmbeddings),
    # Define the output directory for t-SNE results.
    outDir=outputPath / "tSNE_Analysis",
    # Pass the predicted labels for misclassification highlighting.
    predIdxSub=plotPreds,
    # Pass the true labels for ground truth coloring.
    trueIdxSub=plotLabels,
    # Pass the class names from the pipeline.
    customClassNames=dataPipeline.classNames,
    # Enable interactive Plotly HTML export for supplementary materials.
    exportInteractive=True,
    # Enable cluster quality metrics computation.
    enableClusterMetrics=True,
    # Enable misclassification highlighting.
    enableMisclassificationHighlight=True,
    # Enable centroid annotations for cluster identification.
    enableCentroidAnnotations=True,
  )
  # Compute UMAP embeddings and generate publication-ready visualizations using HMB.
  fprint("Computing UMAP and generating enhanced visualizations...")
  # Call the HMB UMAP helper to generate interactive and static plots.
  umapMetrics = UMAPFeaturesExplainability(
    # Pass the subset of embeddings.
    featsSub=plotEmbeddings,
    # Pass None since we provide custom class names.
    labelEncoder=None,
    # Pass the number of samples.
    nSamples=len(plotEmbeddings),
    # Define the output directory for UMAP results.
    outDir=outputPath / "UMAP_Analysis",
    # Pass the predicted labels for misclassification highlighting.
    predIdxSub=plotPreds,
    # Pass the true labels for ground truth coloring.
    trueIdxSub=plotLabels,
    # Pass the class names from the pipeline.
    customClassNames=dataPipeline.classNames,
    # Enable interactive Plotly HTML export for supplementary materials.
    exportInteractive=True,
    # Enable cluster quality metrics computation.
    enableClusterMetrics=True,
    # Enable misclassification highlighting.
    enableMisclassificationHighlight=True,
    # Enable centroid annotations for cluster identification.
    enableCentroidAnnotations=True,
  )
  # Print the completion message.
  fprint("Embedding extraction and advanced latent space visualization complete.")


# Define the main execution block.
if (__name__ == "__main__"):
  # Define the dataset directory.
  datasetDirectory = "./data"
  # Define the model checkpoint path.
  modelCheckpointPath = "./Experiments/BestModel.pt"
  # Define the output directory.
  outputDirectory = "./Experiments/Embeddings"
  # Override the dataset directory with the specific project path.
  datasetDirectory = r"D:\Recent Projects\Intern Projects\Gastric Project\CRC-VAL-HE-7K\SplitDataset"
  # Override the model checkpoint path.
  modelCheckpointPath = r"D:\Recent Projects\Intern Projects\Gastric Project\CRC-VAL-HE-7K\Experiments-CRC-VAL-HE-7K_Output\Exp-EVA02-AdamW-16-CrossEntropy\Seed-42\BestModel.pt"
  # Override the output directory with the specific project path.
  outputDirectory = r"D:\Recent Projects\Intern Projects\Gastric Project\CRC-VAL-HE-7K\Experiments-CRC-VAL-HE-7K_Output\Exp-EVA02-AdamW-16-CrossEntropy\Seed-42\Embeddings"
  # Define the model name.
  modelName = "EVA02"
  # Determine the device to use.
  device = "cuda" if (torch.cuda.is_available()) else "cpu"
  # Define the image size.
  imageSize = 224
  # Define the number of classes.
  numClasses = 9
  # Create the data pipeline to get the dataloaders and class names.
  dataPipeline = PyTorchFolderBasedDataPipeline(dataDir=datasetDirectory, imageSize=imageSize)
  # Build the model.
  model, _ = BuildViTModel(modelName, numClasses, device, imageSize)
  # Load the state dictionary into the model.
  model = LoadModel(model, modelCheckpointPath, device)
  # Move the model to the device.
  model = model.to(device)
  # Set the model to evaluation mode.
  model.eval()
  # Extract and save the embeddings using the custom loop.
  ExtractAndSaveEmbeddings(model, dataPipeline, outputDirectory, device)
