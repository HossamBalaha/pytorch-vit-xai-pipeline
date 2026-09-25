# ==============================================================================
# MNIST VIT IMAGE CLASSIFICATION PIPELINE
# ==============================================================================
# This script runs the full training, validation, and testing pipeline on the
# MNIST dataset using the configurations defined in config.pytorch.yaml.
# ==============================================================================

# Import the operating system module.
import os
# Import the torch module for deep learning operations.
import torch
# Import the yaml module for configuration parsing.
import yaml
# Import the numpy module for numerical operations.
import numpy as np
# Import the Path class from pathlib.
from pathlib import Path
# Import the transforms module from torchvision.
from torchvision import transforms
# Import the MNIST dataset from torchvision.
from torchvision.datasets import MNIST
# Import the DataLoader, Subset, and Dataset utilities from torch.
from torch.utils.data import DataLoader, Subset, Dataset
# Import the train_test_split function from sklearn.
from sklearn.model_selection import train_test_split

# Import the fprint utility from HMB.
from HMB.Utils import fprint
# Import the SeedEverything function from HMB.
from HMB.Initializations import SeedEverything
# Import the HistoryPlotter function from HMB.
from HMB.PerformanceMetrics import HistoryPlotter

# Import core functions from the main pipeline.
from Step1PyTorchPretrainedViTPipeline import (
  LoadConfig,
  EvaluateModel,
  LoadPretrainedViTWeights,
  AdaptModelInputChannels,
  FeatureExtractorWrapper,
)

# Import model building and training utilities.
from HMB.PyTorchClassificationModelsZoo import BuildViTModel
from HMB.PyTorchHelper import (
  EnableMixedPrecision,
  LoadModel,
)
from HMB.PyTorchClassificationLosses import (
  CrossEntropyLossWrapper,
  LabelSmoothingCrossEntropy as LSCE_Loss,
  FocalLossRobust,
)
from HMB.PyTorchTrainingPipeline import TrainEvaluateClassificationModel
from lion_pytorch import Lion
from prodigyopt import Prodigy
from schedulefree import AdamWScheduleFree
from HMB.PyTorchHelper import SophiaG


# Define the MNIST wrapper class to add a dummy filename for evaluation.
class MNISTWrapperEval(Dataset):
  r'''
  Wrapper to make MNIST dataset compatible with EvaluateModel's expectation
  of a 3-tuple output: (image, label, filename).
  Used only for evaluation, NOT for training.
  '''

  # Define the initialization method for the MNIST evaluation wrapper.
  def __init__(self, dataset):
    # Call the parent class initialization method.
    super(MNISTWrapperEval, self).__init__()
    # Store the dataset as an instance variable.
    self.dataset = dataset

  # Define the method to return the length of the dataset.
  def __len__(self):
    # Return the length of the underlying dataset.
    return len(self.dataset)

  # Define the method to get an item from the dataset.
  def __getitem__(self, idx):
    # Get the image and label from the underlying dataset.
    img, label = self.dataset[idx]
    # Generate a dummy filename to satisfy EvaluateModel's unpacking logic.
    filename = f"mnist_img_{idx:05d}.png"
    # Return the image, label, and filename as a tuple.
    return img, label, filename


# Define the function to create and fit the ViT model for MNIST (2-tuple compatible).
def CreateFitViTModelMNIST(
  trainLoader,
  valLoader,
  modelName,
  numClasses,
  numEpochs,
  learningRate,
  device,
  outputDir,
  patience,
  imageSize,
  optimizerName,
  useAmp,
  accumulationSteps,
  useEma,
  useMixup,
  mixupAlpha,
  cutmixAlpha,
  lossFunction,
  labelSmoothing,
  schedulerName,
  judgeBy,
  saveEvery,
  maxGradNorm,
  resumeFromCheckpoint,
  useTopoWassersteinLoss,
  epsilon,
  lambdaTopo,
  lambdaContrastive,
  usePretrainedCustomModels,
  pretrainedModelName,
):
  r'''
  Build and train a ViT model on MNIST using 2-tuple data loaders.
  This function replicates the logic of CreateFitViTModel but handles
  2-tuple batches (data, labels) instead of 3-tuple batches.
  '''

  # Determine effective image size based on model.
  effectiveImageSize = imageSize
  if (modelName == "SwinTransformerV2"):
    effectiveImageSize = 256
  elif (modelName == "SwinTransformerLarge384"):
    effectiveImageSize = 384
  elif (modelName == "EVA02Large"):
    effectiveImageSize = 448

  # Build the ViT model with the specified image size.
  # Note: FeatureExtractorWrapper is omitted because the standard HMB Training Pipeline
  # does not natively support the Topological Wasserstein Loss without custom criterion integration.
  # Wrapping the model causes it to return a tuple (logits, features), which breaks MixupCriterion
  # and standard CrossEntropyLoss.
  model, secondaryModel = BuildViTModel(
    modelName=modelName,
    numClasses=numClasses,
    device=device,
    imageSize=effectiveImageSize,
    usePretrainedCustomModels=usePretrainedCustomModels,
    pretrainedModelName=pretrainedModelName,
  )


  # Determine the input channels from the dataset by peeking at one batch.
  # Use 2-tuple unpacking since MNIST loaders return (data, labels).
  sampleInputs, _ = next(iter(trainLoader))
  targetChannels = sampleInputs.shape[1]

  # Adapt the model's first layer to match the dataset's channel count.
  model = AdaptModelInputChannels(model, targetChannels)

  # Move the model to the specified device.
  model = model.to(device)

  # Define the loss function based on the configuration.
  if (lossFunction == "Focal"):
    # Collect all training labels for class counting.
    trainLabels = []
    for _, labels in trainLoader:
      trainLabels.extend(labels.numpy().tolist())
    trainLabels = np.array(trainLabels)
    classCounts = np.bincount(trainLabels, minlength=numClasses).tolist()
    criterion = FocalLossRobust(numClasses=numClasses, classCounts=classCounts)
  elif (lossFunction == "LabelSmoothing"):
    criterion = LSCE_Loss(labelSmoothing=labelSmoothing)
  else:
    criterion = CrossEntropyLossWrapper()

  # Print a warning if topological loss is requested.
  if (useTopoWassersteinLoss):
    fprint(
      "Warning: Topological Wasserstein Loss is not directly supported by the "
      "standard HMB Training Pipeline wrapper. Using standard loss."
    )

  # Define the model checkpoint path.
  modelCheckpointPath = str(Path(outputDir) / "BestModel.pt")

  # Define the optimizer based on the configuration.
  if (optimizerName == "AdamW"):
    opt = torch.optim.AdamW(model.parameters(), lr=learningRate)
  elif (optimizerName == "SGD"):
    opt = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=0.9)
  elif (optimizerName == "RMSprop"):
    opt = torch.optim.RMSprop(model.parameters(), lr=learningRate)
  elif (optimizerName == "RAdam"):
    opt = torch.optim.RAdam(model.parameters(), lr=learningRate)
  elif (optimizerName == "Lion"):
    opt = Lion(model.parameters(), lr=learningRate)
  elif (optimizerName == "Prodigy"):
    opt = Prodigy(model.parameters(), lr=1.0)
  elif (optimizerName == "ScheduleFreeAdamW"):
    opt = AdamWScheduleFree(model.parameters(), lr=learningRate)
  elif (optimizerName == "Sophia"):
    opt = SophiaG(model.parameters(), lr=learningRate)
  else:
    opt = torch.optim.Adam(model.parameters(), lr=learningRate)

  # Define the learning rate scheduler based on the configuration.
  scheduler = None
  if (schedulerName == "CosineWarmup"):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=numEpochs, eta_min=1e-6)
  elif (schedulerName == "ReduceLROnPlateau"):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

  # Execute the HMB training pipeline with 2-tuple compatible data loaders.
  history = TrainEvaluateClassificationModel(
    model=model,
    criterion=criterion,
    device=torch.device(device),
    bestModelStoragePath=modelCheckpointPath,
    noOfClasses=numClasses,
    numEpochs=numEpochs,
    optimizer=opt,
    scaler=EnableMixedPrecision(model) if useAmp else None,
    scheduler=scheduler,
    trainLoader=trainLoader,
    valLoader=valLoader,
    resumeFromCheckpoint=resumeFromCheckpoint,
    finalModelStoragePath=str(Path(outputDir) / "FinalModel.pt"),
    judgeBy=judgeBy,
    earlyStoppingPatience=patience,
    verbose=True,
    gradAccumSteps=accumulationSteps,
    maxGradNorm=maxGradNorm,
    useAmp=useAmp,
    useMixupFn=useMixup,
    mixUpAlpha=mixupAlpha,
    useEma=useEma,
    saveEvery=saveEvery,
  )

  # Format the history keys for consistency with plotting.
  formattedHistory = {
    "TrainLoss": history.get("train_loss", []),
    "ValLoss"  : history.get("val_loss", []),
    "TrainAcc" : history.get("train_accuracy", []),
    "ValAcc"   : history.get("val_accuracy", []),
  }

  # Load the best model from the checkpoint.
  model = LoadModel(model, modelCheckpointPath, device=device)

  # Build the configuration dictionary for return.
  configs = {
    "ModelName"          : str(modelName),
    "NumClasses"         : int(numClasses),
    "NumEpochs"          : int(numEpochs),
    "LearningRate"       : float(learningRate),
    "OptimizerName"      : str(optimizerName),
    "ModelCheckpointPath": str(modelCheckpointPath),
  }

  # Return the trained model, history, secondary model, and configs.
  return model, formattedHistory, secondaryModel, configs


# Define the function to run the MNIST pipeline.
def RunMNISTPipeline(
  config,
  outputDir,
  modelName,
  numEpochs,
  batchSize,
  imageSize,
  learningRate,
  devicePref,
  patience,
  optimizerName,
  useAmp,
  accumulationSteps,
  useEma,
  useMixup,
  mixupAlpha,
  cutmixAlpha,
  lossFunction,
  labelSmoothing,
  schedulerName,
  useTopoWassersteinLoss,
  epsilon,
  lambdaTopo,
  lambdaContrastive,
  usePretrainedCustomModels,
  pretrainedModelName,
  judgeBy,
  saveEvery,
  maxGradNorm,
  resumeFromCheckpoint
):
  # Print the pipeline start message.
  fprint("Starting MNIST ViT Image Classification Pipeline")

  # Determine the computation device.
  if (devicePref == "auto"):
    # Use CUDA if available, otherwise CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
  else:
    # Use the specified device.
    device = devicePref
  # Print the selected device.
  fprint(f"Using device: {device}")

  # Create the output directory if it doesn't exist.
  Path(outputDir).mkdir(parents=True, exist_ok=True)

  # Define the transform pipeline for MNIST.
  # Resize to target size, convert to 3 channels, and convert to Tensor.
  transformPipeline = transforms.Compose([
    # Resize the image to the target size.
    transforms.Resize((imageSize, imageSize)),
    # Convert grayscale to 3 channels to match Conv2D in_channels=3.
    transforms.Grayscale(num_output_channels=3),
    # Convert the image to a Tensor.
    transforms.ToTensor(),
  ])

  # Print the dataset loading message.
  fprint("Loading MNIST dataset...")
  # Download and load the full MNIST training set.
  trainFull = MNIST(root="./data", train=True, download=True, transform=transformPipeline)
  # Download and load the MNIST test set.
  testDataset = MNIST(root="./data", train=False, download=True, transform=transformPipeline)

  # Split the training set into training and validation sets (90% train, 10% val).
  indices = np.arange(len(trainFull))
  # Perform the train-validation split.
  trainIdx, valIdx = train_test_split(indices, test_size=0.1, random_state=42)

  # Create subsets for training and validation (2-tuple: img, label).
  trainDataset = Subset(trainFull, trainIdx)
  valDataset = Subset(trainFull, valIdx)

  # Print the data loader preparation message.
  fprint("Preparing data loaders...")

  # Create the training DataLoader with 2-tuple batches for TrainOneEpoch compatibility.
  trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True)
  # Create the validation DataLoader with 2-tuple batches for TrainOneEpoch compatibility.
  valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True)

  # Create evaluation DataLoaders with 3-tuple batches for EvaluateModel compatibility.
  # Wrap only the evaluation datasets with the filename-returning wrapper.
  trainDatasetEval = MNISTWrapperEval(trainDataset)
  valDatasetEval = MNISTWrapperEval(valDataset)
  testDatasetEval = MNISTWrapperEval(testDataset)

  trainLoaderEval = DataLoader(trainDatasetEval, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True)
  valLoaderEval = DataLoader(valDatasetEval, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True)
  testLoaderEval = DataLoader(testDatasetEval, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True)

  # Print the data loader ready message.
  fprint("Data loaders ready.")

  # Define the class names for MNIST.
  classNames = [str(i) for i in range(10)]
  # Define the number of classes for MNIST.
  numClasses = 10

  # Execute the training process using the MNIST-compatible function.
  trainedModel, history, secondaryModel, configs = CreateFitViTModelMNIST(
    trainLoader=trainLoader,
    valLoader=valLoader,
    modelName=modelName,
    numClasses=numClasses,
    numEpochs=numEpochs,
    learningRate=learningRate,
    device=device,
    outputDir=outputDir,
    patience=patience,
    imageSize=imageSize,
    optimizerName=optimizerName,
    useAmp=useAmp,
    accumulationSteps=accumulationSteps,
    useEma=useEma,
    useMixup=useMixup,
    mixupAlpha=mixupAlpha,
    cutmixAlpha=cutmixAlpha,
    lossFunction=lossFunction,
    labelSmoothing=labelSmoothing,
    schedulerName=schedulerName,
    judgeBy=judgeBy,
    saveEvery=saveEvery,
    maxGradNorm=maxGradNorm,
    resumeFromCheckpoint=resumeFromCheckpoint,
    useTopoWassersteinLoss=useTopoWassersteinLoss,
    epsilon=epsilon,
    lambdaTopo=lambdaTopo,
    lambdaContrastive=lambdaContrastive,
    usePretrainedCustomModels=usePretrainedCustomModels,
    pretrainedModelName=pretrainedModelName,
  )

  # Print the training completion message.
  fprint(f"Training complete. Best model saved to {configs['ModelCheckpointPath']}")

  # Plot the training history.
  HistoryPlotter(
    history=history,
    title="Training History",
    metrics=("loss", "accuracy"),
    xLabel="Epochs",
    fontSize=14,
    save=True,
    savePath=str(Path(outputDir) / "TrainingHistory.pdf"),
    dpi=720,
    display=False,
    figSize=(14, 5),
    returnFig=False,
    smooth=True,
  )

  # Load the best model from the checkpoint.
  trainedModel.load_state_dict(torch.load(configs["ModelCheckpointPath"]))
  # Move the model to the device.
  trainedModel = trainedModel.to(device)

  # Evaluate on Test, Val, and Train sets using the 3-tuple evaluation loaders.
  for loader, prefix in zip([testLoaderEval, valLoaderEval, trainLoaderEval], ["Test", "Val", "Train"]):
    # Check if the loader is not None.
    if (loader is not None):
      # Perform the evaluation.
      EvaluateModel(
        model=trainedModel,
        dataLoader=loader,
        outputDir=outputDir,
        savePlots=True,
        classNames=classNames,
        prefix=prefix,
        device=device,
        numClasses=numClasses,
        secondaryModel=secondaryModel,
      )

  # Print the pipeline completion message.
  fprint(f"Pipeline Complete | Results saved to {outputDir}")


# Define the main execution block.
if (__name__ == "__main__"):
  # Import the argparse module for command-line interface.
  import argparse

  # Initialize the argument parser for command-line interface.
  parser = argparse.ArgumentParser(description="MNIST ViT Image Classification")
  # Add the command-line argument for the configuration file.
  parser.add_argument("--config", type=str, default="config.pytorch.yaml", help="Path to the YAML configuration file")
  # Parse the command-line arguments.
  args = parser.parse_args()

  # Load the configuration from the file.
  config = LoadConfig(args.config)

  # Extract the base output directory.
  baseOutputDir = str(Path(config.get("OutputDir", "Results")).parent)
  # Extract the batch size from the configuration.
  batchSize = config.get("BatchSize", 16)

  # Extract grid search parameters.
  modelNames = config.get("ModelName", "StandardViT")
  # Extract the list of seeds for multi-trial execution.
  seeds = config.get("Seeds", [42])
  # Extract the optimizers from the configuration.
  optimizers = config.get("Optimizer", "Adam")
  # Extract the loss functions from the configuration.
  lossFunctions = config.get("LossFunction", "CrossEntropy")

  # Extract the pretrained custom models flag.
  usePretrainedCustomModels = config.get("UsePretrainedCustomModels", False)
  # Extract the pretrained model name.
  pretrainedModelName = config.get("PretrainedModelName", "vit_base_patch16_224")
  # Extract the metric to judge the best model by.
  judgeBy = config.get("JudgeBy", "val_accuracy")
  # Extract the save every parameter.
  saveEvery = config.get("SaveEvery", None)
  # Extract the maximum gradient norm.
  maxGradNorm = config.get("MaxGradNorm", None)
  # Extract the resume from checkpoint flag.
  resumeFromCheckpoint = config.get("ResumeFromCheckpoint", False)

  # Ensure parameters are lists for iteration.
  if (isinstance(modelNames, str)):
    # Convert a single string to a list.
    modelNames = [modelNames]
  # Ensure optimizers is a list.
  if (isinstance(optimizers, str)):
    # Convert a single string to a list.
    optimizers = [optimizers]
  # Ensure loss functions is a list.
  if (isinstance(lossFunctions, str)):
    # Convert a single string to a list.
    lossFunctions = [lossFunctions]
  # Ensure seeds is a list.
  if (isinstance(seeds, int)):
    # Convert a single integer seed to a list.
    seeds = [seeds]

  # Loop through all combinations of models, optimizers, loss functions, and seeds.
  for modelName in modelNames:
    # Iterate over each optimizer in the configured list.
    for optimizerName in optimizers:
      # Iterate over each loss function in the configured list.
      for lossFunction in lossFunctions:
        # Iterate over each seed for the multi-trial execution.
        for seed in seeds:
          # Set the global seed for complete reproducibility.
          SeedEverything(seed)
          # Clear the CUDA cache to free up memory before starting a new experiment.
          if (torch.cuda.is_available()):
            # Empty the CUDA cache.
            torch.cuda.empty_cache()

          # Generate a clean model name for the folder path.
          folderModelName = modelName.replace("Transformer", "").replace("ViT", "")
          # Check if the folder model name is Standard.
          if (folderModelName == "Standard"):
            # Set the folder model name to ViT.
            folderModelName = "ViT"

          # Construct dynamic output directory name including the seed.
          experimentName = f"MNIST-Exp-{folderModelName}-{optimizerName}-{batchSize}-{lossFunction}"
          # Construct the current output directory path with the seed subfolder.
          currentOutputDir = str(Path(baseOutputDir) / experimentName / f"Seed-{seed}")

          # Print a separator line for the new experiment.
          fprint(f"\n{'=' * 60}")
          # Print the starting experiment message.
          fprint(f"Starting Experiment: {experimentName} | Seed: {seed}")
          # Print the output directory message.
          fprint(f"Output Directory: {currentOutputDir}")
          # Print a separator line.
          fprint(f"{'=' * 60}\n")

          # Start the try block for experiment execution.
          try:
            # Execute the complete pipeline with parsed configuration parameters.
            RunMNISTPipeline(
              config=config,
              outputDir=currentOutputDir,
              modelName=modelName,
              numEpochs=config.get("NumEpochs", 50),
              batchSize=batchSize,
              imageSize=config.get("ImageSize", 224),
              learningRate=config.get("LearningRate", 1e-4),
              devicePref=config.get("Device", "auto"),
              patience=config.get("Patience", 10),
              optimizerName=optimizerName,
              useAmp=config.get("UseAmp", False),
              accumulationSteps=config.get("AccumulationSteps", 1),
              useEma=config.get("UseEma", False),
              useMixup=config.get("UseMixup", False),
              mixupAlpha=config.get("MixupAlpha", 0.2),
              cutmixAlpha=config.get("CutmixAlpha", 1.0),
              lossFunction=lossFunction,
              labelSmoothing=config.get("LabelSmoothing", 0.0),
              schedulerName=config.get("Scheduler", "None"),
              useTopoWassersteinLoss=config.get("UseTopoWassersteinLoss", False),
              epsilon=config.get("Epsilon", 0.1),
              lambdaTopo=config.get("LambdaTopo", 0.1),
              lambdaContrastive=config.get("LambdaContrastive", 0.1),
              usePretrainedCustomModels=usePretrainedCustomModels,
              pretrainedModelName=pretrainedModelName,
              judgeBy=judgeBy,
              saveEvery=saveEvery,
              maxGradNorm=maxGradNorm,
              resumeFromCheckpoint=resumeFromCheckpoint
            )

            # Create a copy of the configuration for saving.
            configCopy = config.copy()
            # Update the output directory in the configuration copy.
            configCopy["OutputDir"] = currentOutputDir
            # Update the model name in the configuration copy.
            configCopy["ModelName"] = modelName
            # Update the optimizer in the configuration copy.
            configCopy["Optimizer"] = optimizerName
            # Update the loss function in the configuration copy.
            configCopy["LossFunction"] = lossFunction
            # Update the seed in the configuration copy.
            configCopy["Seed"] = seed

            # Open the configuration file for writing.
            with open(Path(currentOutputDir) / "ConfigUsed.yaml", "w") as f:
              # Dump the configuration copy to the YAML file.
              yaml.dump(configCopy, f)

          # Catch any exceptions during the experiment execution.
          except Exception as e:
            # Print the error message.
            fprint(f"ERROR in experiment {experimentName} with seed {seed}: {e}")
            # Import the traceback module for detailed error printing.
            import traceback

            # Print the detailed traceback.
            traceback.print_exc()
            # Print a message indicating continuation.
            fprint("Continuing to the next experiment...")
            # Continue to the next iteration.
            continue

  # Print a separator line for the final metrics.
  fprint("\n" + "=" * 60)
  # Print the completion message.
  fprint("All MNIST experiments completed.")
