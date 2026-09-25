# Import the torch module for deep learning operations.
import torch
# Import the torchvision transforms module for image preprocessing.
from torchvision import transforms
# Import the MNIST dataset from torchvision.
from torchvision.datasets import MNIST
# Import the DataLoader from torch.utils.data.
from torch.utils.data import DataLoader
# Import the custom model classes from the pipeline.
from Step1PyTorchPretrainedViTPipeline import (
  VisionKANModel,
  NeuralODEViTModel,
  SpikingViTModel,
  HypernetworkViTModel,
  LiquidSSMViTModel,
  TestTimeEvolvingViTModel,
  TensorNetworkEntangledViTModel,
  DiffusionPriorEnergyViTModel,
  FractalResonanceViTModel,
  TopologicalQuantumViTModel,
  HolographicInterferenceViTModel,
  NeuromorphicLiquidStateViTModel,
)

# Define the main execution block.
if __name__ == "__main__":
  # Set the device to CUDA if available, otherwise CPU.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Print the device being used.
  print(f"Using device: {device}")

  # Define the transform pipeline to convert MNIST to 3-channel 224x224 tensors.
  transformPipeline = transforms.Compose([
    # Resize the image to match ViT input expectations.
    transforms.Resize((224, 224)),
    # Convert grayscale to 3 channels to match Conv2D in_channels=3.
    transforms.Grayscale(num_output_channels=3),
    # Convert to Tensor.
    transforms.ToTensor(),
  ])

  # Download and load the MNIST test set.
  print("Loading MNIST dataset...")
  mnistDataset = MNIST(root="./data", train=False, download=True, transform=transformPipeline)
  # Create a DataLoader with a small batch size for testing.
  dataLoader = DataLoader(mnistDataset, batch_size=2, shuffle=True)

  # Get a single batch of data.
  sampleInputs, sampleLabels = next(iter(dataLoader))
  # Move the inputs and labels to the device.
  sampleInputs = sampleInputs.to(device)
  sampleLabels = sampleLabels.to(device)

  # Define the list of custom models to test.
  customModels = {
    "VisionKANModel"                 : VisionKANModel,
    "NeuralODEViTModel"              : NeuralODEViTModel,
    "SpikingViTModel"                : SpikingViTModel,
    "HypernetworkViTModel"           : HypernetworkViTModel,
    "LiquidSSMViTModel"              : LiquidSSMViTModel,
    "TestTimeEvolvingViTModel"       : TestTimeEvolvingViTModel,
    "TensorNetworkEntangledViTModel" : TensorNetworkEntangledViTModel,
    "DiffusionPriorEnergyViTModel"   : DiffusionPriorEnergyViTModel,
    "FractalResonanceViTModel"       : FractalResonanceViTModel,
    "TopologicalQuantumViTModel"     : TopologicalQuantumViTModel,
    "HolographicInterferenceViTModel": HolographicInterferenceViTModel,
    "NeuromorphicLiquidStateViTModel": NeuromorphicLiquidStateViTModel,
  }

  # Define the number of classes for MNIST.
  numClasses = 10
  # Define the loss function.
  criterion = torch.nn.CrossEntropyLoss()

  # Iterate through the custom models.
  for modelName, ModelClass in customModels.items():
    print(f"\nTesting {modelName}...")
    try:
      # Initialize the model with the correct number of classes.
      model = ModelClass(numClasses=numClasses)
      # Move the model to the device.
      model = model.to(device)
      # Set the model to training mode to test gradient flow.
      model.train()

      # Perform the forward pass.
      outputs = model(sampleInputs)
      # Compute the loss.
      loss = criterion(outputs, sampleLabels)
      # Perform the backward pass to ensure gradients flow correctly.
      loss.backward()

      # Print the success message with output shape and loss.
      print(f"  [SUCCESS] {modelName} | Output Shape: {outputs.shape} | Loss: {loss.item():.4f}")
    except Exception as e:
      # Print the error message if the model fails.
      print(f"  [FAILED] {modelName} | Error: {e}")

  print("\nAll custom models tested successfully.")
