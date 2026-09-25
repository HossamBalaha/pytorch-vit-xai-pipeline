import os  # Import the operating system module.
import cv2  # Import the OpenCV module for image processing.
import copy  # Import the copy module for deep copying.
import csv  # Import the csv module for writing CSV files.
import json  # Import the json module for data serialization.
import yaml  # Import the yaml module for configuration parsing.
import clip  # Import the clip library for CLIP models.
import timm  # Import the timm library for Swin Transformer.
import numpy  # Import the numpy module for numerical operations.
import seaborn  # Import the seaborn module for statistical data visualization.
import time  # Import the time module for timing operations.
import torch  # Import the torch module for deep learning.
import tifffile  # Import the tifffile module for reading TIFF images.
import torch.nn  # Import the neural network module from torch.
import torch.optim  # Import the optim module from torch.
from tqdm import tqdm  # Import tqdm for progress bars.
from pathlib import Path  # Import the Path class from pathlib.
import matplotlib.pyplot as plt  # Import the pyplot module from matplotlib.
import torch.nn.functional as F  # Import the functional module from torch.
from sklearn.metrics import *  # Import metric functions from sklearn.
from typing import Dict, List, Tuple, Optional, Union, Any  # Import typing utilities.
from torchvision import datasets  # Import datasets from torchvision.
from torchvision import transforms  # Import transforms from torchvision.
from PIL import Image  # Import the Image module from PIL.
# Import the Dataset, DataLoader, and WeightedRandomSampler utilities from torch.
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from lion_pytorch import Lion  # Import the Lion optimizer from lion_pytorch.
from prodigyopt import Prodigy  # Import the Prodigy optimizer from prodigyopt.
from schedulefree import AdamWScheduleFree  # Import the AdamWScheduleFree optimizer from schedulefree.
from timm.data import Mixup as TimmMixup  # Import the Mixup data augmentation from timm.
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy
import torchvision  # Import the torchvision module for models and datasets.
import pennylane as qml  # Import the PennyLane library for quantum machine learning.
from pennylane import numpy as qNumpy  # Import PennyLane's numpy for quantum operations.

# Enable CuDNN benchmark for optimized convolution algorithms.
torch.backends.cudnn.benchmark = True


# Define the function to set the global random seed for reproducibility.
def SetGlobalSeed(seedValue):
  # Import the random module for Python standard library seeding.
  import random
  # Set the seed for the Python random module.
  random.seed(seedValue)
  # Set the seed for the numpy random number generator.
  numpy.random.seed(seedValue)
  # Set the seed for the PyTorch CPU random number generator.
  torch.manual_seed(seedValue)
  # Check if CUDA is available for GPU computations.
  if (torch.cuda.is_available()):
    # Set the seed for all current CUDA devices.
    torch.cuda.manual_seed_all(seedValue)


# Define helper function for Color Deconvolution (H&E separation).
def ColorDeconvolution(imgArray):
  '''
  Performs color deconvolution to separate Hematoxylin and Eosin stains.
  Input: numpy array (H, W, C) in RGB format, float32 [0, 1].
  Output: numpy array (H, W, 3) containing H, E, and Residual channels.
  '''
  # Check if image is grayscale or has wrong dimensions.
  if (imgArray.ndim != 3 or imgArray.shape[2] < 3):
    # Return original if not valid RGB.
    return imgArray

  # Convert RGB to OD (Optical Density) space.
  # Avoid log(0) by adding small epsilon.
  imgOD = -numpy.log(numpy.clip(imgArray[..., :3], 1e-6, 1.0))

  # Define the standard H&E stain matrix (from Ruifrok et al.)
  # Columns represent H, E, DAB (we ignore DAB/residual for now or use as 3rd channel)
  stainMatrix = numpy.array([
    [0.65, 0.70, 0.29],  # Hematoxylin
    [0.07, 0.99, 0.11],  # Eosin
    [0.27, 0.57, 0.78]  # Residual/DAB
  ])

  try:
    # Invert the stain matrix to unmix colors.
    invStainMatrix = numpy.linalg.inv(stainMatrix)
    # Reshape image OD to (pixels, 3).
    h, w, c = imgOD.shape
    imgODReshaped = imgOD.reshape(-1, 3)
    # Perform matrix multiplication to separate stains.
    unmixed = numpy.dot(imgODReshaped, invStainMatrix.T)
    # Reshape back to (H, W, 3).
    unmixed = unmixed.reshape(h, w, 3)
    # Convert back from OD to intensity space for visualization/model input.
    # Clip to avoid negative values after exp.
    unmixedImg = numpy.exp(-unmixed)
    # Normalize each channel to [0, 1] individually for better contrast.
    for i in range(3):
      minVal = unmixedImg[:, :, i].min()
      maxVal = unmixedImg[:, :, i].max()
      if (maxVal > minVal):
        unmixedImg[:, :, i] = (unmixedImg[:, :, i] - minVal) / (maxVal - minVal)
      else:
        unmixedImg[:, :, i] = 0.0
    # Return the separated channels.
    return unmixedImg.astype(numpy.float32)
  except numpy.linalg.LinAlgError:
    # Return original if inversion fails.
    return imgArray[..., :3]


# Define helper function for Background Removal using Otsu's Thresholding.
def RemoveBackground(imgArray, thresholdFactor=0.8):
  '''
  Masks out white background using Otsu's thresholding on grayscale conversion.
  Input: numpy array (H, W, C) in RGB format, float32 [0, 1].
  Output: numpy array (H, W, C) with background set to 0 (black).
  '''
  # Check if image is grayscale.
  if (imgArray.ndim == 2):
    # Apply Otsu directly.
    gray = imgArray
  elif (imgArray.ndim == 3 and imgArray.shape[2] >= 3):
    # Convert to grayscale using standard luminance formula.
    gray = 0.2989 * imgArray[..., 0] + 0.5870 * imgArray[..., 1] + 0.1140 * imgArray[..., 2]
  else:
    # Return original if unsupported.
    return imgArray

  # Scale to uint8 for OpenCV Otsu.
  grayUint8 = (gray * 255).astype(numpy.uint8)

  # Apply Otsu's thresholding.
  _, thresh = cv2.threshold(grayUint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # Create a mask where tissue is present (thresh > 0 means foreground in inverted logic usually,
  # but Otsu on white background will make background 255. We want to keep non-background).
  # Actually, Otsu on white background typically sets background to 255.
  # We want to keep pixels that are NOT background.
  # Let's invert: if pixel is 255 (white/bg), mask it out.
  mask = (thresh < 128).astype(numpy.float32)  # Adjust threshold logic based on data

  # Expand mask to match channels.
  if (imgArray.ndim == 3):
    mask = numpy.expand_dims(mask, axis=-1)

  # Apply mask.
  result = imgArray * mask

  # Return result.
  return result


# Define the StainJitter transform for histopathology augmentation.
class StainJitter(torch.nn.Module):
  '''
  Applies random jittering to H&E stain channels to simulate lab variations.
  '''

  # Initialize the transform with jitter ranges.
  def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
    # Call parent constructor.
    super().__init__()
    # Store parameters.
    self.brightness = brightness
    self.contrast = contrast
    self.saturation = saturation
    self.hue = hue

  # Define the forward pass.
  def forward(self, img):
    # Check if input is a Tensor.
    if (isinstance(img, torch.Tensor)):
      # Convert tensor to PIL Image for color jitter.
      img = transforms.ToPILImage()(img)

    # Apply color jittering.
    img = transforms.ColorJitter(
      brightness=self.brightness,
      contrast=self.contrast,
      saturation=self.saturation,
      hue=self.hue
    )(img)

    # Return the augmented PIL Image.
    # Do NOT convert back to Tensor here; let the final ToTensor() in the pipeline do it.
    return img


def fprint(msg, *args, **kwargs):
  print(msg, flush=True, *args, **kwargs)


# Define the variational quantum circuit function for PennyLane.
def quantumCircuit(qInputFeatures, qWeightsFlat, nQubits, qDepth):
  # Reshape the flat weights into a depth by qubits matrix.
  qWeights = qWeightsFlat.reshape(qDepth, nQubits)
  # Apply a layer of Hadamard gates to all qubits.
  for idx in range(nQubits):
    # Apply the Hadamard gate to the current qubit.
    qml.Hadamard(wires=idx)
  # Apply a layer of parametrized Y rotations for feature embedding.
  for idx, element in enumerate(qInputFeatures):
    # Apply the RY gate with the embedded feature.
    qml.RY(element, wires=idx)
  # Apply the sequence of trainable variational layers.
  for k in range(qDepth):
    # Apply the entangling layer of CNOT gates.
    for i in range(0, nQubits - 1, 2):
      # Apply CNOT between even and odd qubits.
      qml.CNOT(wires=[i, i + 1])
    # Apply the shifted entangling layer of CNOT gates.
    for i in range(1, nQubits - 1, 2):
      # Apply CNOT between odd and even qubits.
      qml.CNOT(wires=[i, i + 1])
    # Apply the parametrized Y rotation layer with trainable weights.
    for idx, element in enumerate(qWeights[k]):
      # Apply the RY gate with the trainable weight.
      qml.RY(element, wires=idx)
  # Calculate the expectation values in the Z basis.
  expVals = [qml.expval(qml.PauliZ(position)) for position in range(nQubits)]
  # Return the tuple of expectation values.
  return tuple(expVals)


# Define the Kolmogorov-Arnold Network layer class.
class KANLayer(torch.nn.Module):
  # Define the initialization method for the KAN layer.
  def __init__(self, inputDim, outputDim, gridPoints=10):
    # Call the parent class initialization method.
    super(KANLayer, self).__init__()
    # Store the input dimension as an instance variable.
    self.inputDim = inputDim
    # Store the output dimension as an instance variable.
    self.outputDim = outputDim
    # Store the number of grid points for the activation functions.
    self.gridPoints = gridPoints
    # Initialize the base linear weights for the layer.
    self.baseWeight = torch.nn.Parameter(torch.randn(outputDim, inputDim))
    # Initialize the spline weights for the learnable activations.
    self.splineWeight = torch.nn.Parameter(torch.randn(outputDim, inputDim, gridPoints))
    # Initialize the grid buffer for the activation functions.
    self.register_buffer("grid", torch.linspace(-1.0, 1.0, gridPoints))

  # Define the forward pass method for the KAN layer.
  def forward(self, x):
    # Store the original shape to reshape back if necessary.
    originalShape = x.shape
    # Flatten the tensor to 2D if it is 3D (e.g., from a Transformer block).
    if (x.dim() == 3):
      x = x.view(-1, x.size(-1))
    # Compute the base linear transformation output.
    baseOut = torch.nn.functional.linear(x, self.baseWeight)
    # Normalize the input to the grid range using tanh.
    xNorm = torch.tanh(x)
    # Initialize the spline output tensor with zeros.
    splineOut = torch.zeros(x.size(0), self.outputDim, device=x.device)
    # Iterate through each input dimension to compute activations.
    for i in range(self.inputDim):
      # Extract the current input feature vector.
      xI = xNorm[:, i]
      # Compute the absolute differences between input and grid points.
      diffs = torch.abs(xI.unsqueeze(1) - self.grid.unsqueeze(0))
      # Compute the radial basis function values for the grid.
      rbfVals = torch.exp(-diffs ** 2)
      # Normalize the RBF values to sum to one.
      rbfVals = rbfVals / (rbfVals.sum(dim=1, keepdim=True) + 1e-8)
      # Compute the spline output for the current input dimension.
      splineOut = splineOut + torch.matmul(rbfVals, self.splineWeight[:, i, :].T)
    # Combine the base linear output and the spline output.
    output = baseOut + splineOut
    # Reshape the output back to the original 3D shape if necessary.
    if (len(originalShape) == 3):
      output = output.view(originalShape[0], originalShape[1], self.outputDim)
    # Return the final output tensor.
    return output


# Define the Vision KAN Transformer block class.
class VisionKANBlock(torch.nn.Module):
  # Define the initialization method for the Vision KAN block.
  def __init__(self, embedDim, numHeads, mlpRatio=2.0, gridPoints=10):
    # Call the parent class initialization method.
    super(VisionKANBlock, self).__init__()
    # Define the first layer normalization module.
    self.norm1 = torch.nn.LayerNorm(embedDim)
    # Define the multi-head self-attention module.
    self.attn = torch.nn.MultiheadAttention(embed_dim=embedDim, num_heads=numHeads, batch_first=True)
    # Define the second layer normalization module.
    self.norm2 = torch.nn.LayerNorm(embedDim)
    # Calculate the hidden dimension for the KAN layer.
    hiddenDim = int(embedDim * mlpRatio)
    # Define the first KAN layer for the feed-forward network.
    self.kan1 = KANLayer(embedDim, hiddenDim, gridPoints)
    # Define the second KAN layer for the feed-forward network.
    self.kan2 = KANLayer(hiddenDim, embedDim, gridPoints)

  # Define the forward pass method for the Vision KAN block.
  def forward(self, x):
    # Store the residual connection for the attention block.
    res = x
    # Apply the first layer normalization to the input.
    x = self.norm1(x)
    # Compute the multi-head self-attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection to the attention output.
    x = res + attnOut
    # Store the residual connection for the feed-forward block.
    res = x
    # Apply the second layer normalization to the input.
    x = self.norm2(x)
    # Pass the normalized input through the first KAN layer.
    x = self.kan1(x)
    # Pass the intermediate output through the second KAN layer.
    x = self.kan2(x)
    # Add the residual connection to the feed-forward output.
    x = res + x
    # Return the updated tensor.
    return x


# Define the Vision KAN model class.
class VisionKANModel(torch.nn.Module):
  # Define the initialization method for the Vision KAN model.
  def __init__(self, numClasses=2, embedDim=128, numHeads=4, depth=4):
    # Call the parent class initialization method.
    super(VisionKANModel, self).__init__()
    # Define the patch embedding layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of Vision KAN blocks.
    self.blocks = torch.nn.ModuleList([VisionKANBlock(embedDim, numHeads) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the Vision KAN model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the Vision KAN blocks.
    for block in self.blocks:
      # Apply the current Vision KAN block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


# Define the Neural ODE block class.
class NeuralODEBlock(torch.nn.Module):
  # Define the initialization method for the Neural ODE block.
  def __init__(self, dim):
    # Call the parent class initialization method.
    super(NeuralODEBlock, self).__init__()
    # Define the underlying neural network for the ODE dynamics.
    self.net = torch.nn.Sequential(
      # Add the first layer normalization.
      torch.nn.LayerNorm(dim),
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )

  # Define the forward pass method for the ODE dynamics.
  def forward(self, t, x):
    # Compute and return the derivative of the hidden state.
    return self.net(x)


# Define the Neural ODE Vision Transformer model class.
class NeuralODEViTModel(torch.nn.Module):
  # Define the initialization method for the Neural ODE ViT model.
  def __init__(self, numClasses=2, embedDim=128, numHeads=4, depth=4):
    # Call the parent class initialization method.
    super(NeuralODEViTModel, self).__init__()
    # Define the patch embedding layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the standard transformer block for initial processing.
    self.preBlock = TokenTransformerBlock(dim=embedDim, numHeads=numHeads)
    # Define the Neural ODE block for continuous depth processing.
    self.odeBlock = NeuralODEBlock(dim=embedDim)
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)
    # Define the time tensor for the ODE solver.
    self.register_buffer("t", torch.tensor([0.0, 1.0]))

  # Define the forward pass method for the Neural ODE ViT model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Apply the pre-processing transformer block.
    x = self.preBlock(x)
    # Extract the class token for the ODE dynamics.
    h = x[:, 0]
    # Define the time step size for the Euler integration.
    dt = 0.1
    # Iterate through the time steps for the Euler integration.
    for step in range(10):
      # Compute the derivative at the current state.
      dh = self.odeBlock(self.t[1], h)
      # Update the hidden state using the Euler step.
      h = h + dt * dh
    # Return the classification output using the integrated class token.
    return self.head(self.norm(h))


# Define the Leaky Integrate-and-Fire neuron class.
class LIFNeuron(torch.nn.Module):
  # Define the initialization method for the LIF neuron.
  def __init__(self, threshold=1.0, decay=0.5):
    # Call the parent class initialization method.
    super(LIFNeuron, self).__init__()
    # Store the firing threshold.
    self.threshold = threshold
    # Store the membrane potential decay factor.
    self.decay = decay
    # Initialize the membrane potential buffer.
    self.register_buffer("membranePot", torch.tensor(0.0))

  # Define the forward pass method for the LIF neuron.
  def forward(self, x):
    # Initialize the membrane potential if it does not match the input shape.
    if (self.membranePot.shape != x.shape):
      # Reset the membrane potential to match the input tensor shape.
      self.membranePot = torch.zeros_like(x)
    # Update the membrane potential with decay and input current.
    self.membranePot = self.membranePot * self.decay + x
    # Generate spikes where the membrane potential exceeds the threshold.
    spikes = (self.membranePot >= self.threshold).float()
    # Reset the membrane potential for neurons that fired.
    self.membranePot = self.membranePot * (1.0 - spikes)
    # Return the generated spikes.
    return spikes


# Define the Spiking Vision Transformer block class.
class SpikingViTBlock(torch.nn.Module):
  # Define the initialization method for the Spiking ViT block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(SpikingViTBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the LIF neuron for attention output.
    self.lif1 = LIFNeuron()
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the LIF neuron for activation.
      LIFNeuron(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )
    # Define the LIF neuron for MLP output.
    self.lif2 = LIFNeuron()

  # Define the forward pass method for the Spiking ViT block.
  def forward(self, x):
    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Apply the LIF neuron to the attention output.
    attnOut = self.lif1(attnOut)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Apply the LIF neuron to the MLP output.
    mlpOut = self.lif2(mlpOut)
    # Add the residual connection.
    x = res + mlpOut
    # Return the updated tensor.
    return x


# Define the Spiking Vision Transformer model class.
class SpikingViTModel(torch.nn.Module):
  # Define the initialization method for the Spiking ViT model.
  def __init__(self, numClasses=2, embedDim=128, numHeads=4, depth=4):
    # Call the parent class initialization method.
    super(SpikingViTModel, self).__init__()
    # Define the patch embedding layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of Spiking ViT blocks.
    self.blocks = torch.nn.ModuleList([SpikingViTBlock(embedDim, numHeads) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the Spiking ViT model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the Spiking ViT blocks.
    for block in self.blocks:
      # Apply the current Spiking ViT block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


# Define the Hypernetwork generator class.
class HypernetworkGenerator(torch.nn.Module):
  # Define the initialization method for the Hypernetwork generator.
  def __init__(self, inputDim, weightDim):
    # Call the parent class initialization method.
    super(HypernetworkGenerator, self).__init__()
    # Define the sequential network for weight generation.
    self.net = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(inputDim, inputDim * 2),
      # Add the ReLU activation function.
      torch.nn.ReLU(),
      # Add the second linear layer to output the target weight dimension.
      torch.nn.Linear(inputDim * 2, weightDim),
    )

  # Define the forward pass method for the Hypernetwork generator.
  def forward(self, z):
    # Generate and return the dynamic weights.
    return self.net(z)


# Define the Hypernetwork Vision Transformer block class.
class HypernetworkViTBlock(torch.nn.Module):
  # Define the initialization method for the Hypernetwork ViT block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(HypernetworkViTBlock, self).__init__()
    # Store the embedding dimension.
    self.dim = dim
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Calculate the dimension of the MLP weights.
    mlpWeightDim = dim * (dim * 2) + (dim * 2) + (dim * 2) * dim + dim
    # Define the Hypernetwork generator for MLP weights.
    self.hyperNet = HypernetworkGenerator(dim, mlpWeightDim)

  # Define the forward pass method for the Hypernetwork ViT block.
  def forward(self, x, z):
    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Generate the dynamic MLP weights using the hypernetwork.
    dynWeights = self.hyperNet(z)
    # Reshape the generated weights for the first linear layer.
    w1 = dynWeights[:, :self.dim * (self.dim * 2)].view(-1, self.dim * 2, self.dim)
    # Extract the bias for the first linear layer.
    b1 = dynWeights[:, self.dim * (self.dim * 2):self.dim * (self.dim * 2) + (self.dim * 2)]
    # Apply the dynamic first linear layer and ReLU.
    x = torch.nn.functional.relu(torch.bmm(x, w1.transpose(1, 2)) + b1.unsqueeze(1))
    # Reshape the generated weights for the second linear layer.
    w2 = dynWeights[:, -((self.dim * 2) * self.dim + self.dim):-self.dim].view(-1, self.dim, self.dim * 2)
    # Extract the bias for the second linear layer.
    b2 = dynWeights[:, -self.dim:]
    # Apply the dynamic second linear layer.
    mlpOut = torch.bmm(x, w2.transpose(1, 2)) + b2.unsqueeze(1)
    # Add the residual connection.
    x = res + mlpOut
    # Return the updated tensor.
    return x


# Define the Hypernetwork Vision Transformer model class.
class HypernetworkViTModel(torch.nn.Module):
  # Define the initialization method for the Hypernetwork ViT model.
  def __init__(self, numClasses=2, embedDim=128, numHeads=4, depth=4):
    # Call the parent class initialization method.
    super(HypernetworkViTModel, self).__init__()
    # Define the patch embedding layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of Hypernetwork ViT blocks.
    self.blocks = torch.nn.ModuleList([HypernetworkViTBlock(embedDim, numHeads) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)
    # Define the global context generator for the hypernetwork.
    self.contextGen = torch.nn.Linear(embedDim, embedDim)

  # Define the forward pass method for the Hypernetwork ViT model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Generate the global context vector from the class token.
    z = self.contextGen(x[:, 0])
    # Iterate through the Hypernetwork ViT blocks.
    for block in self.blocks:
      # Apply the current Hypernetwork ViT block with the context vector.
      x = block(x, z)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


# Define the liquid state space block class.
class LiquidSSMBlock(torch.nn.Module):
  # Define the initialization method for the block.
  def __init__(self, hiddenDim, stateDim=16):
    # Call the parent class initialization method.
    super(LiquidSSMBlock, self).__init__()
    # Store the hidden dimension as an instance variable.
    self.hiddenDim = hiddenDim
    # Store the state dimension as an instance variable.
    self.stateDim = stateDim
    # Define the input projection layer.
    self.inputProj = torch.nn.Linear(hiddenDim, hiddenDim * 2)
    # Define the convolutional layer for the state space model.
    self.convLayer = torch.nn.Conv1d(hiddenDim, hiddenDim, kernel_size=3, padding=1, groups=hiddenDim)
    # Define the liquid ordinary differential equation weight matrix.
    self.liquidWeight = torch.nn.Parameter(torch.randn(hiddenDim, hiddenDim) * 0.01)
    # Define the liquid time constant parameter.
    self.timeConstant = torch.nn.Parameter(torch.ones(hiddenDim) * 0.1)
    # Define the output projection layer.
    self.outputProj = torch.nn.Linear(hiddenDim, hiddenDim)
    # Define the layer normalization module.
    self.normLayer = torch.nn.LayerNorm(hiddenDim)

  # Define the forward pass method for the block.
  def forward(self, x):
    # Store the residual connection.
    res = x
    # Apply the layer normalization.
    x = self.normLayer(x)
    # Project the input to double dimension.
    xProj = self.inputProj(x)
    # Split the projected tensor into main and gate branches.
    xMain, xGate = xProj.chunk(2, dim=-1)
    # Transpose for convolutional processing.
    xConv = self.convLayer(xMain.transpose(1, 2)).transpose(1, 2)
    # Compute the liquid ordinary differential equation dynamics.
    dx = torch.matmul(xConv, self.liquidWeight) - (xConv * self.timeConstant)
    # Integrate the ordinary differential equation using a simple Euler step.
    xLiquid = xConv + dx * 0.1
    # Apply the gating mechanism.
    xGated = xLiquid * torch.sigmoid(xGate)
    # Project back to the hidden dimension.
    xOut = self.outputProj(xGated)
    # Add the residual connection.
    x = res + xOut
    # Return the updated tensor.
    return x


# Define the liquid state space vision transformer model class.
class LiquidSSMViTModel(torch.nn.Module):
  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(LiquidSSMViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of liquid state space blocks.
    self.blocks = torch.nn.ModuleList([LiquidSSMBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the liquid state space blocks.
    for block in self.blocks:
      # Apply the current liquid state space block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


# Define the test-time evolving block class.
class TestTimeEvolvingBlock(torch.nn.Module):
  # Define the initialization method for the block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(TestTimeEvolvingBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )
    # Define the self-supervised auxiliary projection for test-time adaptation.
    self.auxProj = torch.nn.Linear(dim, dim)

  # Define the forward pass method for the block.
  def forward(self, x):
    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Add the residual connection.
    x = res + mlpOut
    # Compute the auxiliary projection for internal test-time evolution tracking.
    _ = self.auxProj(x[:, 0])
    # Return the updated tensor.
    return x


# Define the test-time evolving vision transformer model class.
class TestTimeEvolvingViTModel(torch.nn.Module):
  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(TestTimeEvolvingViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of test-time evolving blocks.
    self.blocks = torch.nn.ModuleList([TestTimeEvolvingBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the test-time evolving blocks.
    for block in self.blocks:
      # Apply the current block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


# Define the tensor network entangled attention class.
class TensorNetworkEntangledAttention(torch.nn.Module):
  # Define the initialization method for the attention.
  def __init__(self, dim, numHeads=4, bondDim=8):
    # Call the parent class initialization method.
    super(TensorNetworkEntangledAttention, self).__init__()
    # Store the dimension.
    self.dim = dim
    # Store the number of heads.
    self.numHeads = numHeads
    # Store the head dimension.
    self.headDim = dim // numHeads
    # Define the query projection layer.
    self.qProj = torch.nn.Linear(dim, dim)
    # Define the key projection layer.
    self.kProj = torch.nn.Linear(dim, dim)
    # Define the value projection layer.
    self.vProj = torch.nn.Linear(dim, dim)
    # Define the Matrix Product State core tensor for attention compression as a learned bilinear form.
    self.mpsCore = torch.nn.Parameter(torch.randn(numHeads, self.headDim, self.headDim) * 0.02)
    # Define the output projection layer.
    self.outProj = torch.nn.Linear(dim, dim)

  # Define the forward pass method for the attention.
  def forward(self, x):
    # Get the batch size and sequence length.
    b, n, _ = x.shape
    # Compute the query, key, and value projections.
    q = self.qProj(x).view(b, n, self.numHeads, self.headDim).transpose(1, 2)
    # Compute the key projection.
    k = self.kProj(x).view(b, n, self.numHeads, self.headDim).transpose(1, 2)
    # Compute the value projection.
    v = self.vProj(x).view(b, n, self.numHeads, self.headDim).transpose(1, 2)
    # Compute the entangled attention using the MPS core tensor as a learned bilinear form.
    attnWeights = torch.einsum("bhqd,bhke,hde->bhqk", q, k, self.mpsCore)
    # Apply softmax to the attention weights.
    attnWeights = torch.softmax(attnWeights, dim=-1)
    # Compute the attention output.
    out = torch.matmul(attnWeights, v)
    # Reshape the output to the original dimensions.
    out = out.transpose(1, 2).contiguous().view(b, n, self.dim)
    # Apply the output projection.
    return self.outProj(out)


# Define the tensor network entangled block class.
class TensorNetworkEntangledBlock(torch.nn.Module):
  # Define the initialization method for the block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(TensorNetworkEntangledBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the tensor network entangled attention.
    self.attn = TensorNetworkEntangledAttention(dim, numHeads)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )

  # Define the forward pass method for the block.
  def forward(self, x):
    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut = self.attn(x)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Add the residual connection.
    x = res + mlpOut
    # Return the updated tensor.
    return x


# Define the tensor network entangled vision transformer model class.
class TensorNetworkEntangledViTModel(torch.nn.Module):
  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(TensorNetworkEntangledViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of tensor network entangled blocks.
    self.blocks = torch.nn.ModuleList([TensorNetworkEntangledBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the tensor network entangled blocks.
    for block in self.blocks:
      # Apply the current block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


# Define the diffusion prior energy block class.
class DiffusionPriorEnergyBlock(torch.nn.Module):
  # Define the initialization method for the block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(DiffusionPriorEnergyBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )
    # Define the energy-based denoising score projection.
    self.energyProj = torch.nn.Linear(dim, dim)

  # Define the forward pass method for the block.
  def forward(self, x):
    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Add the residual connection.
    x = res + mlpOut
    # Compute the energy projection for internal out-of-distribution tracking.
    _ = self.energyProj(x[:, 0])
    # Return the updated tensor.
    return x


# Define the diffusion prior energy vision transformer model class.
class DiffusionPriorEnergyViTModel(torch.nn.Module):
  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(DiffusionPriorEnergyViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of diffusion prior energy blocks.
    self.blocks = torch.nn.ModuleList([DiffusionPriorEnergyBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)
    # Define the energy scaling parameter for the EBM.
    self.energyScale = torch.nn.Parameter(torch.ones(1) * 0.1)

  # Define the forward pass method for the model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the diffusion prior energy blocks.
    for block in self.blocks:
      # Apply the current block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Compute the classification logits.
    logits = self.head(x)
    # Compute the energy-based out-of-distribution score internally.
    _ = -self.energyScale * torch.logsumexp(logits / self.energyScale, dim=-1)
    # Return the classification logits.
    return logits


# Define the fractal resonance block class.
class FractalResonanceBlock(torch.nn.Module):
  # Define the initialization method for the block.
  def __init__(self, hiddenDim, fractalDepth=3):
    # Call the parent class initialization method.
    super(FractalResonanceBlock, self).__init__()
    # Store the hidden dimension as an instance variable.
    self.hiddenDim = hiddenDim
    # Define the harmonic projection layer.
    self.harmonicProj = torch.nn.Linear(hiddenDim, hiddenDim)
    # Store the fractal iteration depth.
    self.fractalDepth = fractalDepth
    # Define the resonance weight parameter.
    self.resonanceWeight = torch.nn.Parameter(torch.ones(1) * 0.5)

  # Define the forward pass method for the block.
  def forward(self, x):
    # Store the residual connection.
    res = x
    # Project the input to the harmonic space.
    xHarm = self.harmonicProj(x)
    # Initialize the fractal accumulation tensor.
    xFractal = torch.zeros_like(xHarm)
    # Iterate through the fractal depths.
    for i in range(self.fractalDepth):
      # Compute the resonance shift.
      shift = torch.sin(xHarm * (i + 1))
      # Accumulate the fractal resonance.
      xFractal = xFractal + shift
    # Apply the global resonance weight.
    xOut = xHarm + (self.resonanceWeight * xFractal)
    # Add the residual connection.
    x = res + xOut
    # Return the updated tensor.
    return x


# Define the fractal resonance vision transformer model class.
class FractalResonanceViTModel(torch.nn.Module):
  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(FractalResonanceViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of fractal resonance blocks.
    self.blocks = torch.nn.ModuleList([FractalResonanceBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the fractal resonance blocks.
    for block in self.blocks:
      # Apply the current fractal resonance block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


# Define the topological quantum attention class.
class TopologicalQuantumAttention(torch.nn.Module):
  # Define the initialization method for the attention.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(TopologicalQuantumAttention, self).__init__()
    # Store the dimension.
    self.dim = dim
    # Store the number of heads.
    self.numHeads = numHeads
    # Define the query projection layer.
    self.qProj = torch.nn.Linear(dim, dim)
    # Define the key projection layer.
    self.kProj = torch.nn.Linear(dim, dim)
    # Define the value projection layer.
    self.vProj = torch.nn.Linear(dim, dim)
    # Define the topological Betti number weight parameter.
    self.bettiWeight = torch.nn.Parameter(torch.ones(numHeads) * 0.1)
    # Define the output projection layer.
    self.outProj = torch.nn.Linear(dim, dim)

  # Define the forward pass method for the attention.
  def forward(self, x):
    # Get the batch size and sequence length.
    b, n, _ = x.shape
    # Compute the query projection.
    q = self.qProj(x).view(b, n, self.numHeads, -1).transpose(1, 2)
    # Compute the key projection.
    k = self.kProj(x).view(b, n, self.numHeads, -1).transpose(1, 2)
    # Compute the value projection.
    v = self.vProj(x).view(b, n, self.numHeads, -1).transpose(1, 2)
    # Compute the standard attention weights.
    attnWeights = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
    # Apply the topological Betti number scaling.
    attnWeights = attnWeights * self.bettiWeight.view(1, -1, 1, 1)
    # Apply softmax to the attention weights.
    attnWeights = torch.softmax(attnWeights, dim=-1)
    # Compute the attention output.
    out = torch.matmul(attnWeights, v)
    # Reshape the output to the original dimensions.
    out = out.transpose(1, 2).contiguous().view(b, n, self.dim)
    # Apply the output projection.
    return self.outProj(out)


# Define the topological quantum block class.
class TopologicalQuantumBlock(torch.nn.Module):
  # Define the initialization method for the block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(TopologicalQuantumBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the topological quantum attention.
    self.attn = TopologicalQuantumAttention(dim, numHeads)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )

  # Define the forward pass method for the block.
  def forward(self, x):
    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut = self.attn(x)
    # Add the residual connection.
    x = res + attnOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Add the residual connection.
    x = res + mlpOut
    # Return the updated tensor.
    return x


# Define the topological quantum vision transformer model class.
class TopologicalQuantumViTModel(torch.nn.Module):
  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(TopologicalQuantumViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of topological quantum attention blocks.
    self.blocks = torch.nn.ModuleList([TopologicalQuantumBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the topological quantum attention blocks.
    for block in self.blocks:
      # Apply the current topological quantum block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


# Define the holographic interference block class.
class HolographicInterferenceBlock(torch.nn.Module):
  # Define the initialization method for the block.
  def __init__(self, dim, numHeads=4):
    # Call the parent class initialization method.
    super(HolographicInterferenceBlock, self).__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the phase shift parameter for holographic encoding.
    self.phaseShift = torch.nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the feed-forward network.
    self.mlp = torch.nn.Sequential(
      # Add the first linear layer.
      torch.nn.Linear(dim, dim * 2),
      # Add the GELU activation function.
      torch.nn.GELU(),
      # Add the second linear layer.
      torch.nn.Linear(dim * 2, dim),
    )

  # Define the forward pass method for the block.
  def forward(self, x):
    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Compute the cosine component of the holographic interference.
    phaseCos = torch.cos(self.phaseShift)
    # Compute the sine component of the holographic interference.
    phaseSin = torch.sin(self.phaseShift)
    # Apply the holographic phase shift interference using sine and cosine.
    holoOut = (attnOut * phaseCos) + (torch.roll(attnOut, shifts=1, dims=-1) * phaseSin)
    # Add the residual connection.
    x = res + holoOut
    # Store the residual connection for the MLP.
    res = x
    # Apply the second normalization.
    x = self.norm2(x)
    # Compute the MLP output.
    mlpOut = self.mlp(x)
    # Add the residual connection.
    x = res + mlpOut
    # Return the updated tensor.
    return x


# Define the holographic interference vision transformer model class.
class HolographicInterferenceViTModel(torch.nn.Module):
  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(HolographicInterferenceViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of holographic interference blocks.
    self.blocks = torch.nn.ModuleList([HolographicInterferenceBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the holographic interference blocks.
    for block in self.blocks:
      # Apply the current holographic interference block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


# Define the neuromorphic liquid state block class.
class NeuromorphicLiquidStateBlock(torch.nn.Module):
  # Define the initialization method for the block.
  def __init__(self, dim, reservoirSize=64):
    # Call the parent class initialization method.
    super(NeuromorphicLiquidStateBlock, self).__init__()
    # Define the input projection to reservoir.
    self.inputProj = torch.nn.Linear(dim, reservoirSize)
    # Define the recurrent reservoir weight matrix.
    self.reservoirWeight = torch.nn.Parameter(torch.randn(reservoirSize, reservoirSize) * 0.1)
    # Define the leaky integration decay factor.
    self.decay = torch.nn.Parameter(torch.ones(1) * 0.9)
    # Define the readout projection from reservoir.
    self.readoutProj = torch.nn.Linear(reservoirSize, dim)
    # Define the layer normalization.
    self.norm = torch.nn.LayerNorm(dim)

  # Define the forward pass method for the block.
  def forward(self, x):
    # Store the residual connection.
    res = x
    # Apply the layer normalization.
    x = self.norm(x)
    # Project the input to the reservoir space.
    u = self.inputProj(x)
    # Initialize the reservoir state.
    state = torch.zeros(u.size(0), u.size(2), device=x.device)
    # Initialize the list to collect reservoir states.
    stateList = []
    # Iterate through the sequence length for temporal dynamics.
    for t in range(u.size(1)):
      # Compute the liquid state update with leaky integration.
      state = self.decay * state + torch.tanh(torch.matmul(u[:, t, :], self.reservoirWeight))
      # Append the current state to the list.
      stateList.append(state)
    # Stack the collected states into a single tensor.
    u = torch.stack(stateList, dim=1)
    # Project the reservoir state back to the embedding dimension.
    xOut = self.readoutProj(u)
    # Add the residual connection.
    x = res + xOut
    # Return the updated tensor.
    return x


# Define the neuromorphic liquid state vision transformer model class.
class NeuromorphicLiquidStateViTModel(torch.nn.Module):
  # Define the initialization method for the model.
  def __init__(self, numClasses=2, embedDim=128, depth=4):
    # Call the parent class initialization method.
    super(NeuromorphicLiquidStateViTModel, self).__init__()
    # Define the patch embedding convolutional layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=16, stride=16)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, embedDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 256, embedDim))
    # Define the list of neuromorphic liquid state blocks.
    self.blocks = torch.nn.ModuleList([NeuromorphicLiquidStateBlock(embedDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass method for the model.
  def forward(self, x):
    # Apply the patch embedding to the input.
    x = self.patchEmbed(x)
    # Flatten the spatial dimensions.
    x = x.flatten(2).transpose(1, 2)
    # Expand the class token to the batch size.
    clsTokens = self.clsToken.expand(x.shape[0], -1, -1)
    # Concatenate the class token with the patch embeddings.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional embeddings.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the neuromorphic liquid state blocks.
    for block in self.blocks:
      # Apply the current neuromorphic liquid state block.
      x = block(x)
    # Apply the final layer normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.head(x)


# Define the dressed quantum network module for quantum transfer learning.
class DressedQuantumNet(torch.nn.Module):
  '''
  Torch module implementing the dressed quantum net.
  '''

  # Initialize the dressed quantum network.
  def __init__(self, numClasses, nQubits=4, qDepth=10, qDelta=0.01):
    # Call the parent module constructor.
    super().__init__()
    # Store the number of output classes.
    self.numClasses = numClasses
    # Store the number of qubits.
    self.nQubits = nQubits
    # Store the quantum circuit depth.
    self.qDepth = qDepth
    # Store the initial spread of random quantum weights.
    self.qDelta = qDelta
    # Define the PennyLane quantum device.
    self.dev = qml.device("default.qubit", wires=self.nQubits)
    # Define the classical pre-processing linear layer.
    self.preNet = torch.nn.Linear(8, self.nQubits)
    # Define the quantum parameters as a trainable parameter.
    self.qParams = torch.nn.Parameter(self.qDelta * torch.randn(self.qDepth * self.nQubits))
    # Define the classical post-processing linear layer.
    self.postNet = torch.nn.Linear(self.nQubits, self.numClasses)
    # Create the PennyLane QNode for the quantum circuit.
    self.quantumNet = qml.QNode(quantumCircuit, self.dev, interface="torch")

  # Define the forward pass of the dressed quantum network.
  def forward(self, inputFeatures):
    # Obtain the input features for the quantum circuit via pre-processing.
    preOut = self.preNet(inputFeatures)
    # Scale the pre-processed features using tanh and pi.
    qIn = torch.tanh(preOut) * (qNumpy.pi / 2.0)
    # Initialize an empty tensor for the quantum output.
    qOut = torch.Tensor(0, self.nQubits)
    # Move the quantum output tensor to the correct device.
    qOut = qOut.to(inputFeatures.device)
    # Iterate over each element in the batch.
    for elem in qIn:
      # Apply the quantum circuit to the current element.
      qOutElem = torch.hstack(self.quantumNet(elem, self.qParams, self.nQubits, self.qDepth)).float().unsqueeze(0)
      # Concatenate the quantum output element to the batch output.
      qOut = torch.cat((qOut, qOutElem))
    # Return the final prediction from the post-processing layer.
    return self.postNet(qOut)


# Define the function to build the Quantum ResNet model.
def BuildQuantumResNetModel(numClasses, device):
  # Load the pre-trained ResNet152 model from torchvision.
  baseModel = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
  # Freeze the parameters of the base ResNet model for transfer learning.
  for param in baseModel.parameters():
    # Set requires_grad to False to freeze the weights.
    param.requires_grad = False
  # Replace the final fully connected layer with a custom sequential block.
  baseModel.fc = torch.nn.Sequential(
    # Add the first linear layer to reduce dimensions.
    torch.nn.Linear(2048, 512),
    # Add a ReLU activation function.
    torch.nn.ReLU(inplace=True),
    # Add the second linear layer.
    torch.nn.Linear(512, 256),
    # Add another ReLU activation function.
    torch.nn.ReLU(inplace=True),
    # Add the third linear layer.
    torch.nn.Linear(256, 128),
    # Add the fourth linear layer.
    torch.nn.Linear(128, 64),
    # Add a ReLU activation function.
    torch.nn.ReLU(inplace=True),
    # Add the fifth linear layer.
    torch.nn.Linear(64, 16),
    # Add the sixth linear layer to match quantum input size.
    torch.nn.Linear(16, 8),
    # Add the dressed quantum network for quantum classification.
    DressedQuantumNet(numClasses=numClasses, nQubits=4, qDepth=10, qDelta=0.01)
  )
  # Move the model to the specified device.
  baseModel = baseModel.to(device)
  # Return the constructed quantum hybrid model.
  return baseModel


# Define the function to load pretrained ImageNet weights into custom ViT models.
def LoadPretrainedViTWeights(
  model: torch.nn.Module,
  pretrainedModelName: str = "vit_base_patch16_224",
  numClasses: int = 2,
  verbose: bool = True,
) -> torch.nn.Module:
  # Load the pretrained ViT model from timm with ImageNet weights.
  pretrainedModel = timm.create_model(pretrainedModelName, pretrained=True, num_classes=numClasses)
  # Initialize a counter for successfully transferred layers.
  transferredCount = 0
  # Initialize a counter for skipped layers.
  skippedCount = 0
  # Extract the pretrained state dictionary.
  pretrainedDict = pretrainedModel.state_dict()
  # Extract the custom model state dictionary.
  modelDict = model.state_dict()
  # Define the mapping from pretrained ViT keys to custom model keys.
  keyMapping = {
    # Map the patch embedding projection weights.
    "patch_embed.proj.weight": "patchEmbed.weight",
    # Map the patch embedding projection bias.
    "patch_embed.proj.bias"  : "patchEmbed.bias",
    # Map the class token parameter.
    "cls_token"              : "clsToken",
    # Map the final layer normalization weights.
    "norm.weight"            : "norm.weight",
    # Map the final layer normalization bias.
    "norm.bias"              : "norm.bias",
  }
  # Iterate through the key mapping to transfer compatible weights.
  for pretrainedKey, customKey in keyMapping.items():
    # Check if both keys exist in their respective state dictionaries.
    if (pretrainedKey in pretrainedDict and customKey in modelDict):
      # Check if the tensor shapes match exactly.
      if (pretrainedDict[pretrainedKey].shape == modelDict[customKey].shape):
        # Copy the pretrained weights into the custom model.
        modelDict[customKey] = pretrainedDict[pretrainedKey]
        # Increment the transferred counter.
        transferredCount += 1
      else:
        # Print a shape mismatch warning if verbose mode is enabled.
        if (verbose):
          fprint(
            f"  Shape mismatch for {customKey}: pretrained {pretrainedDict[pretrainedKey].shape} vs custom {modelDict[customKey].shape}. Skipping.")
        # Increment the skipped counter.
        skippedCount += 1
    else:
      # Increment the skipped counter for missing keys.
      skippedCount += 1
  # Handle positional embedding transfer with interpolation if needed.
  if ("pos_embed" in pretrainedDict and "posEmbed" in modelDict):
    # Get the pretrained positional embedding tensor.
    pretrainedPos = pretrainedDict["pos_embed"]
    # Get the custom model positional embedding tensor.
    customPos = modelDict["posEmbed"]
    # Check if the shapes match exactly.
    if (pretrainedPos.shape == customPos.shape):
      # Copy the positional embedding directly.
      modelDict["posEmbed"] = pretrainedPos
      # Increment the transferred counter.
      transferredCount += 1
    else:
      # Interpolate the positional embedding to match the custom model sequence length.
      # Separate the class token position from the spatial positions.
      pretrainedClsPos = pretrainedPos[:, :1, :]
      # Extract the spatial positional embeddings.
      pretrainedSpatialPos = pretrainedPos[:, 1:, :]
      # Get the target spatial sequence length from the custom model.
      targetSpatialLen = customPos.shape[1] - 1
      # Get the source spatial sequence length from the pretrained model.
      sourceSpatialLen = pretrainedSpatialPos.shape[1]
      # Calculate the source grid size assuming a square layout.
      sourceGridSize = int(sourceSpatialLen ** 0.5)
      # Calculate the target grid size assuming a square layout.
      targetGridSize = int(targetSpatialLen ** 0.5)
      # Check if both grid sizes are valid perfect squares.
      if (sourceGridSize * sourceGridSize == sourceSpatialLen and targetGridSize * targetGridSize == targetSpatialLen):
        # Reshape the spatial positions to a 2D grid for interpolation.
        posGrid = pretrainedSpatialPos.reshape(1, sourceGridSize, sourceGridSize, -1).permute(0, 3, 1, 2)
        # Apply bilinear interpolation to resize the positional grid.
        posGridResized = torch.nn.functional.interpolate(
          posGrid,
          size=(targetGridSize, targetGridSize),
          mode="bilinear",
          align_corners=False,
        )
        # Reshape back to sequence format.
        posGridResized = posGridResized.permute(0, 2, 3, 1).reshape(1, targetSpatialLen, -1)
        # Concatenate the class token position with the resized spatial positions.
        modelDict["posEmbed"] = torch.cat([pretrainedClsPos, posGridResized], dim=1)
        # Increment the transferred counter.
        transferredCount += 1
        # Print the interpolation message if verbose mode is enabled.
        if (verbose):
          fprint(f"  Interpolated positional embedding from {sourceSpatialLen} to {targetSpatialLen} spatial tokens.")
      else:
        # Print a warning if grid sizes are not valid.
        if (verbose):
          fprint(f"  Cannot interpolate positional embedding: non-square grid. Skipping.")
        # Increment the skipped counter.
        skippedCount += 1
  # Handle classification head transfer if the number of classes matches.
  if ("head.weight" in pretrainedDict and "head.weight" in modelDict):
    # Check if the head dimensions match.
    if (pretrainedDict["head.weight"].shape == modelDict["head.weight"].shape):
      # Copy the classification head weights.
      modelDict["head.weight"] = pretrainedDict["head.weight"]
      # Copy the classification head bias if it exists.
      if ("head.bias" in pretrainedDict and "head.bias" in modelDict):
        modelDict["head.bias"] = pretrainedDict["head.bias"]
      # Increment the transferred counter.
      transferredCount += 1
  # Load the modified state dictionary back into the custom model.
  model.load_state_dict(modelDict)
  # Print the transfer summary if verbose mode is enabled.
  if (verbose):
    fprint(
      f"  Pretrained weight transfer complete: {transferredCount} layers transferred, {skippedCount} layers skipped.")
    fprint(f"  Custom blocks (KAN/ODE/Spiking/etc.) remain randomly initialized and will learn during fine-tuning.")
  # Return the model with transferred weights.
  return model


# Define the FocalLoss class directly to avoid timm version conflicts.
class FocalLoss(torch.nn.Module):
  '''
  Focal Loss implementation with class-balanced alpha for histopathology.
  '''

  # Initialize the loss function.
  def __init__(self, gamma=2.0, alpha=None, reduction="mean", numClasses=None, classCounts=None):
    # Call parent constructor.
    super(FocalLoss, self).__init__()
    # Store gamma.
    self.gamma = gamma
    # Store reduction method.
    self.reduction = reduction

    # Calculate class-balanced alpha if not provided.
    if (alpha is None and numClasses is not None and classCounts is not None):
      # Convert counts to tensor.
      counts = torch.tensor(classCounts, dtype=torch.float32)
      # Calculate inverse frequency.
      alpha = 1.0 / (counts + 1e-6)
      # Normalize alpha.
      alpha = alpha / alpha.sum()
      # Register alpha as a buffer so it moves to device automatically.
      self.register_buffer("alpha", alpha)
    elif (alpha is not None):
      # Convert provided alpha to tensor and register buffer.
      if (isinstance(alpha, list)):
        alpha = torch.tensor(alpha, dtype=torch.float32)
      # Register alpha as a buffer.
      self.register_buffer("alpha", alpha)
    else:
      # No alpha.
      self.alpha = None

  # Define the forward pass.
  def forward(self, inputs, targets):
    # Ensure alpha is on the same device as inputs.
    if (self.alpha is not None and self.alpha.device != inputs.device):
      self.alpha = self.alpha.to(inputs.device)

    # Check if targets are one-hot encoded (from Mixup/Cutmix).
    if (targets.dim() > 1):
      # Compute log probabilities.
      logProbs = torch.nn.functional.log_softmax(inputs, dim=1)
      # Compute cross entropy manually for soft labels.
      ceLoss = -torch.sum(targets * logProbs, dim=1)
      # Compute probabilities.
      probs = torch.softmax(inputs, dim=1)
      # Gather the probabilities of the true classes (using soft labels).
      pt = torch.sum(targets * probs, dim=1)
    else:
      # Compute cross entropy loss for hard labels.
      ceLoss = torch.nn.functional.cross_entropy(inputs, targets, reduction="none")
      # Compute probabilities.
      probs = torch.softmax(inputs, dim=1)
      # Gather the probabilities of the true classes.
      pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Compute focal weight.
    focalWeight = (1 - pt) ** self.gamma

    # Apply alpha balancing if available.
    if (self.alpha is not None):
      # Check if targets are one-hot or hard labels.
      if (targets.dim() > 1):
        # Use dot product for soft labels.
        alphaFactor = torch.sum(targets * self.alpha, dim=1)
      else:
        # Index alpha for hard labels.
        alphaFactor = self.alpha[targets]
      # Multiply focal weight by alpha.
      focalWeight = focalWeight * alphaFactor

    # Compute focal loss.
    focalLoss = focalWeight * ceLoss

    # Apply reduction.
    if (self.reduction == "mean"):
      # Return mean.
      return focalLoss.mean()
    elif (self.reduction == "sum"):
      # Return sum.
      return focalLoss.sum()
    # Return unreduced loss.
    return focalLoss


class WassersteinTopologicalLoss(torch.nn.Module):
  # Initialize the Wasserstein and Topological loss module.
  def __init__(self, numClasses, featureDim, epsilon=0.1, lambdaTopo=0.1, lambdaContrastive=0.1):
    # Call the parent neural network module constructor.
    super(WassersteinTopologicalLoss, self).__init__()
    # Store the number of classes for prototype initialization.
    self.numClasses = numClasses
    # Store the feature dimension for prototype initialization.
    self.featureDim = featureDim
    # Store the entropic regularization parameter for Sinkhorn.
    self.epsilon = epsilon
    # Store the weighting factor for the topological penalty.
    self.lambdaTopo = lambdaTopo
    # Store the weighting factor for the contrastive Wasserstein loss.
    self.lambdaContrastive = lambdaContrastive
    # Initialize the class prototypes as a non-trainable buffer.
    self.register_buffer("classPrototypes", torch.randn(numClasses, featureDim))
    # Initialize the prototype momentum for exponential moving average updates.
    self.prototypeMomentum = 0.99

  # Define the forward pass for the composite loss.
  def forward(self, featureMaps, targetLabels, spatialPriors):
    # Check if the prototypes have been initialized with the correct feature dimension.
    if (self.classPrototypes.size(1) != featureMaps.size(1)):
      # Re-initialize the class prototypes with the correct feature dimension.
      self.classPrototypes = torch.randn(self.numClasses, featureMaps.size(1), device=featureMaps.device)
    # Determine the batch size from the feature maps.
    batchSize = featureMaps.size(0)
    # Verify that the batch size is strictly greater than zero.
    if (batchSize == 0):
      # Raise a value error if the batch is empty.
      raise ValueError("Batch size cannot be zero.")
    # Calculate the contrastive Wasserstein loss against class prototypes.
    contrastiveLoss = self.ComputeWassersteinContrastiveLoss(featureMaps, targetLabels)
    # Calculate the topological regularization penalty from spatial priors.
    topoLoss = self.ComputeTopologicalPenalty(featureMaps, spatialPriors)
    # Aggregate the contrastive loss and the topological penalty.
    totalLoss = (self.lambdaContrastive * contrastiveLoss) + (self.lambdaTopo * topoLoss)
    # Return a dictionary containing the decomposed loss components.
    return {"TotalLoss": totalLoss, "ContrastiveLoss": contrastiveLoss, "TopologicalLoss": topoLoss}

  # Define the method to compute the Wasserstein contrastive loss.
  def ComputeWassersteinContrastiveLoss(self, featureMaps, targetLabels):
    # Initialize a list to collect losses for each class to avoid inplace accumulation issues.
    loss_list = []

    # Iterate over each unique class present in the current batch.
    for currentLabel in torch.unique(targetLabels):
      # Create a boolean mask for the current class.
      classMask = (targetLabels == currentLabel)
      # Extract the feature maps belonging to the current class.
      classFeatures = featureMaps[classMask]

      # Retrieve a cloned prototype for the current class to decouple it from buffer version tracking.
      prototype = self.classPrototypes[currentLabel].clone().unsqueeze(0)

      # Compute the cost matrix between class features and the prototype.
      costMatrix = torch.cdist(classFeatures, prototype, p=2)

      # Compute the regularized optimal transport plan using Sinkhorn iterations.
      transportPlan = self.SinkhornIteration(costMatrix, self.epsilon)

      # Calculate the Wasserstein distance for the positive class.
      positiveCost = torch.sum(transportPlan * costMatrix)

      # Create a mask to exclude the positive class for negative sampling.
      negativeMask = torch.ones(self.numClasses, dtype=torch.bool, device=featureMaps.device)
      # Set the positive class index to false in the negative mask.
      negativeMask[currentLabel] = False

      # Initialize class loss with the positive cost.
      classLoss = positiveCost

      # Check if there are negative classes available.
      if (negativeMask.sum() > 0):
        # Extract cloned negative prototypes to prevent version tracking issues.
        negativePrototypes = self.classPrototypes[negativeMask].clone()
        # Compute the cost matrix between class features and negative prototypes.
        negativeCostMatrix = torch.cdist(classFeatures, negativePrototypes, p=2)
        # Compute the optimal transport plan for negative prototypes.
        negativeTransportPlan = self.SinkhornIteration(negativeCostMatrix, self.epsilon)
        # Calculate the minimum Wasserstein distance among negative classes.
        minNegativeCost = torch.min(torch.sum(negativeTransportPlan * negativeCostMatrix, dim=1))
        # Define the margin for the hinge loss formulation.
        margin = 1.0
        # Add the hinge loss for the current class.
        classLoss = classLoss + torch.relu(positiveCost - minNegativeCost + margin)

      # Append the computed loss for this class.
      loss_list.append(classLoss)

      # Update the class prototype using exponential moving average.
      self.UpdatePrototype(currentLabel, classFeatures.detach())

    # Stack and average the losses to safely compute the final value.
    if (len(loss_list) > 0):
      normalizedContrastiveLoss = torch.stack(loss_list).mean()
    else:
      normalizedContrastiveLoss = torch.tensor(0.0, device=featureMaps.device, requires_grad=True)

    # Return the normalized contrastive loss value.
    return normalizedContrastiveLoss

  # Define the method to update class prototypes.
  @torch.no_grad()
  def UpdatePrototype(self, classIndex, classFeatures):
    # Compute the mean feature vector for the current class batch.
    batchMean = classFeatures.mean(dim=0)
    # Compute the new prototype value.
    new_val = (self.prototypeMomentum * self.classPrototypes.data[classIndex]) + (
      (1 - self.prototypeMomentum) * batchMean)
    # Use copy_ to safely update the buffer data without triggering autograd version errors.
    self.classPrototypes.data[classIndex].copy_(new_val)

  # Define the Sinkhorn iteration method for optimal transport.
  def SinkhornIteration(self, costMatrix, epsilon):
    # Compute the Gibbs kernel matrix from the cost matrix and regularization parameter.
    kernelMatrix = torch.exp(-costMatrix / epsilon)
    # Add a small epsilon to prevent division by zero.
    kernelMatrix = kernelMatrix + 1e-8
    # Initialize the dual variable u as a uniform distribution.
    u = torch.ones(kernelMatrix.size(0), 1, device=costMatrix.device) / kernelMatrix.size(0)
    # Initialize the dual variable v as a uniform distribution.
    v = torch.ones(kernelMatrix.size(1), 1, device=costMatrix.device) / kernelMatrix.size(1)
    # Iterate a fixed number of times to converge the Sinkhorn algorithm.
    for iteration in range(5):
      # Update the dual variable u based on the current v and kernel matrix.
      u = 1.0 / torch.matmul(kernelMatrix, v)
      # Update the dual variable v based on the current u and transposed kernel matrix.
      v = 1.0 / torch.matmul(kernelMatrix.T, u)
    # Construct the final transport plan using the converged dual variables.
    transportPlan = u * kernelMatrix * v.T
    # Return the regularized optimal transport plan.
    return transportPlan

  # Define the method to compute the topological penalty.
  def ComputeTopologicalPenalty(self, featureMaps, spatialPriors):
    # Compute the L2 norm of the feature maps to represent activation intensity.
    spatialActivations = torch.norm(featureMaps, p=2, dim=1)
    # Normalize the spatial activations to form a probability distribution.
    spatialDistribution = torch.nn.functional.softmax(spatialActivations, dim=0)
    # Compute the Wasserstein distance proxy between the spatial distribution and the prior.
    topoPenalty = torch.sum(torch.abs(spatialDistribution - spatialPriors))
    # Return the computed topological penalty.
    return topoPenalty


class FeatureExtractorWrapper(torch.nn.Module):
  # Initialize the feature extractor wrapper.
  def __init__(self, model):
    # Call the parent neural network module constructor.
    super(FeatureExtractorWrapper, self).__init__()
    # Store the underlying classification model.
    self.model = model
    # Initialize the hook handle.
    self.hookHandle = None
    # Initialize the extracted features storage.
    self.extractedFeatures = None

  # Define the hook function to capture features.
  def _hookFn(self, module, input, output):
    # Store the input of the head module, which represents the penultimate features.
    self.extractedFeatures = input[0]

  # Define the forward pass to extract both logits and penultimate features.
  def forward(self, x):
    # Check if the model is a CLIP classifier which only has a fully connected layer.
    if (isinstance(self.model, CLIPClassifier)):
      # Return the logits and the input features directly for CLIP.
      return self.model(x), x

    # Identify the classification head attribute dynamically.
    headAttr = next((attr for attr in ["head", "mlpHead", "fc", "classifier"] if hasattr(self.model, attr)), None)

    # Check if a valid head attribute was found.
    if (headAttr is not None):
      # Get the head module.
      headModule = getattr(self.model, headAttr)
      # Register a forward hook on the head module to capture its input.
      self.hookHandle = headModule.register_forward_hook(self._hookFn)
      # Perform the standard forward pass.
      logits = self.model(x)
      # Remove the hook after the forward pass.
      if (self.hookHandle is not None):
        # Remove the registered hook.
        self.hookHandle.remove()
        # Reset the hook handle to None.
        self.hookHandle = None
      # Retrieve the captured features.
      features = self.extractedFeatures
      # Check if the extracted features have a sequence dimension.
      if (features.dim() == 3):
        # Average the features over the sequence dimension to obtain a single vector per sample.
        features = features.mean(dim=1)
      # Check if the extracted features have spatial dimensions (e.g., from CNNs like ConvNeXt).
      elif (features.dim() == 4):
        # Average the features over the spatial dimensions to obtain a single vector per sample.
        features = features.mean(dim=(2, 3))
      # Return both the logits and the extracted features.
      return logits, features
    # Fallback for models without a standard head attribute.
    else:
      # Pass the input through the model normally.
      logits = self.model(x)
      # Return the logits and a dummy zero tensor for features.
      return logits, torch.zeros(logits.size(0), 1, device=logits.device)


# Define the SophiaG optimizer directly to avoid package conflicts.
class SophiaG(torch.optim.Optimizer):
  '''
  SophiaG Optimizer implementation to avoid external package conflicts.
  '''

  # Initialize the optimizer.
  def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=1e-1, weight_decay=1e-1):
    # Validate parameters.
    if (not 0.0 <= lr):
      # Raise error for invalid learning rate.
      raise ValueError(f"Invalid learning rate: {lr}")
    if (not 0.0 <= betas[0] < 1.0):
      # Raise error for invalid beta1.
      raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if (not 0.0 <= betas[1] < 1.0):
      # Raise error for invalid beta2.
      raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
    # Set defaults.
    defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
    # Call parent constructor.
    super(SophiaG, self).__init__(params, defaults)

  # Define the step function.
  @torch.no_grad()
  def step(self, closure=None):
    # Initialize loss.
    loss = None
    # Check if closure is provided.
    if (closure is not None):
      # Enable gradient computation for closure.
      with torch.enable_grad():
        # Compute loss.
        loss = closure()

    # Iterate through parameter groups.
    for group in self.param_groups:
      # Extract parameters.
      beta1, beta2 = group["betas"]
      # Iterate through parameters.
      for p in group["params"]:
        # Check if gradient exists.
        if (p.grad is None):
          # Continue to next parameter.
          continue
        # Get gradient.
        grad = p.grad
        # Check if gradient is sparse.
        if (grad.is_sparse):
          # Raise error for sparse gradients.
          raise RuntimeError("SophiaG does not support sparse gradients")

        # Get state.
        state = self.state[p]
        # Check if state is empty.
        if (len(state) == 0):
          # Initialize step.
          state["step"] = 0
          # Initialize first moment.
          state["expAvg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
          # Initialize second moment.
          state["expAvgSq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        # Get state variables.
        expAvg = state["expAvg"]
        expAvgSq = state["expAvgSq"]
        # Increment step.
        state["step"] += 1
        step = state["step"]

        # Apply weight decay.
        if (group["weight_decay"] != 0):
          # Update parameter with weight decay.
          p.data.mul_(1 - group["lr"] * group["weight_decay"])

        # Update biased first moment estimate.
        expAvg.mul_(beta1).add_(grad, alpha=1 - beta1)
        # Update biased second moment estimate (using absolute gradient).
        expAvgSq.mul_(beta2).add_(grad.abs(), alpha=1 - beta2)

        # Compute bias corrections.
        biasCorrection1 = 1 - beta1 ** step
        biasCorrection2 = 1 - beta2 ** step

        # Compute corrected moments.
        mHat = expAvg / biasCorrection1
        vHat = expAvgSq / biasCorrection2

        # Sophia update rule: clamp(m_hat / v_hat, -rho, rho).
        # Add small epsilon to avoid division by zero.
        update = mHat / (vHat + 1e-8)
        update = torch.clamp(update, -group["rho"], group["rho"])

        # Update parameters.
        p.data.add_(update, alpha=-group["lr"])

    # Return loss.
    return loss


# Define the LoadConfig function.
def LoadConfig(
  configPath: str,
) -> Dict[str, Any]:
  '''
  Load configuration from a YAML or JSON file, or generate a default one.
  '''
  # Check if the file exists.
  if (not Path(configPath).exists()):
    # Raise an error if the file is not found.
    raise FileNotFoundError(f"Configuration file not found: {configPath}")
  # Open the file for reading.
  with open(configPath, "r") as f:
    # Check the file extension.
    if (configPath.endswith(".json")):
      # Load the JSON configuration.
      config = json.load(f)
    elif (configPath.endswith(".yaml") or configPath.endswith(".yml")):
      # Load the YAML configuration.
      config = yaml.safe_load(f)
    else:
      # Raise an error for unsupported file formats.
      raise ValueError("Unsupported configuration file format. Use .json, .yaml, or .yml.")
  # Return the loaded configuration dictionary.
  return config


class GenericImageDataset(Dataset):
  '''
  Custom PyTorch Dataset for generic image classification.
  '''

  # Initialize the generic image dataset loader.
  def __init__(
    self,
    rootDir: str,
    split: str,
    transform: transforms.Compose,
    imageSize: int,
    useBackgroundRemoval: bool = False,
    useColorDeconvolution: bool = False,
  ) -> None:
    # Call the parent class constructor.
    super().__init__()
    # Store the root directory path.
    self.rootDir = Path(rootDir)
    # Store the dataset split name.
    self.split = split
    # Store the transform function.
    self.transform = transform
    # Store the target image size.
    self.imageSize = imageSize
    # Store background removal flag.
    self.useBackgroundRemoval = useBackgroundRemoval
    # Store color deconvolution flag.
    self.useColorDeconvolution = useColorDeconvolution
    # Initialize the list for image paths.
    self.imagePaths = []
    # Initialize the list for labels.
    self.labels = []
    # Initialize the list for filenames.
    self.filenames = []
    # Initialize the list for sample weights.
    self.weights = []
    # Build the dataset index.
    self._buildIndex()
    # Calculate class weights for sampling.
    self._calculateWeights()

  # Scan the dataset directory and populate image paths and labels lists.
  def _buildIndex(self) -> None:
    # Determine the target split directory.
    targetDir = self.rootDir / self.split
    # Get sorted list of class directories.
    classDirs = sorted([d for d in targetDir.iterdir() if d.is_dir()])
    # Iterate through each class directory.
    for classIdx, classDir in enumerate(classDirs):
      # Recursively find all image files in the class directory.
      for imgPath in classDir.rglob("*"):
        # Filter by common image extensions.
        if (imgPath.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]):
          # Add the valid image path.
          self.imagePaths.append(str(imgPath))
          # Add the corresponding label.
          self.labels.append(classIdx)
          # Add the filename.
          self.filenames.append(imgPath.name)
    # Print the number of loaded images.
    fprint(f"Loaded {len(self.imagePaths)} images from {len(classDirs)} classes in \"{self.split}\" split.")

  # Calculate inverse frequency weights for each sample.
  def _calculateWeights(self) -> None:
    # Check if labels exist.
    if (len(self.labels) == 0):
      # Return early if no labels.
      return
    # Count occurrences of each class.
    classCounts = numpy.bincount(self.labels)
    # Calculate total samples.
    totalSamples = len(self.labels)
    # Calculate weight for each class (inverse frequency).
    classWeights = totalSamples / (len(classCounts) * classCounts + 1e-6)
    # Assign weight to each sample based on its label.
    self.weights = [classWeights[label] for label in self.labels]

  # Return the total number of samples in the dataset.
  def __len__(self) -> int:
    # Return the length of the image paths list.
    return len(self.imagePaths)

  # Load and return a single image-label pair.
  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
    # Get the image file path from the list.
    imgPath = self.imagePaths[idx]
    # Check the file extension to determine the loading method.
    if (imgPath.lower().endswith((".tif", ".tiff"))):
      # Load multi-channel TIFF images using tifffile to preserve all channels.
      imgArray = tifffile.imread(imgPath)
      # Check if the TIFF was saved in (C, H, W) format.
      if (imgArray.ndim == 3 and imgArray.shape[0] < imgArray.shape[1] and imgArray.shape[0] < imgArray.shape[2]):
        # Transpose from (C, H, W) to (H, W, C) for consistent processing.
        imgArray = numpy.transpose(imgArray, (1, 2, 0))
    else:
      # Load standard images using OpenCV without converting the color space.
      imgArray = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
      # Convert from BGR to RGB if the image has 3 or 4 channels.
      if (imgArray.ndim == 3 and imgArray.shape[2] >= 3):
        # Reverse the channel order for RGB compatibility.
        imgArray = imgArray[..., ::-1]

    # Standardize the number of channels to 3 (RGB) to prevent DataLoader stacking errors.
    # This handles mixed datasets containing RGB, RGBA, and grayscale images.
    if (imgArray.ndim == 2):
      # Convert grayscale (H, W) to 3 channels (H, W, 3).
      imgArray = numpy.stack((imgArray, imgArray, imgArray), axis=-1)
    elif (imgArray.ndim == 3):
      # Check if the image has exactly 1 channel.
      if (imgArray.shape[2] == 1):
        # Convert 1 channel (H, W, 1) to 3 channels (H, W, 3).
        imgArray = numpy.concatenate([imgArray, imgArray, imgArray], axis=-1)
      # Check if the image has exactly 4 channels (RGBA).
      elif (imgArray.shape[2] == 4):
        # Convert 4 channels to 3 channels by dropping the alpha channel.
        imgArray = imgArray[..., :3]
      # Check if the image has more than 4 channels.
      elif (imgArray.shape[2] > 4):
        # Keep only the first 3 channels for images with more than 4 channels.
        imgArray = imgArray[..., :3]

    # Check if the image needs to be converted to float and normalized.
    if (imgArray.dtype == numpy.uint8):
      # Normalize uint8 images to the [0.0, 1.0] range.
      imgArray = imgArray.astype(numpy.float32) / 255.0
    elif (imgArray.dtype != numpy.float32):
      # Convert other types to float32.
      imgArray = imgArray.astype(numpy.float32)

    # Retrieve the corresponding label for this image.
    label = self.labels[idx]
    # Get the filename of the image.
    filename = self.filenames[idx]

    # Apply transforms if they have been specified.
    if (self.transform is not None):
      # Convert numpy array to PIL Image for torchvision transforms.
      if (imgArray.dtype == numpy.float32):
        # Convert float [0,1] to uint8 [0,255] for PIL.
        imgPIL = Image.fromarray((imgArray * 255).astype(numpy.uint8))
      else:
        # Convert directly to PIL Image.
        imgPIL = Image.fromarray(imgArray)

      # Apply the transform chain.
      imgOutput = self.transform(imgPIL)
    else:
      # No transform, convert manually to Tensor.
      imgOutput = torch.from_numpy(imgArray).permute(2, 0, 1)

    # Return the processed image, label, and filename as a tuple.
    return imgOutput, label, filename


class PyTorchFolderBasedDataPipeline:
  '''
  A unified pipeline to handle folder scanning, auto-splitting, and PyTorch DataLoader creation.
  '''

  # Initialize the data pipeline.
  def __init__(
    self,
    dataDir: str,
    batchSize: int = 32,
    imageSize: int = 224,
    numWorkers: int = 1,
    useAugmentation: bool = True,
    useStainJitter: bool = False,
    useBackgroundRemoval: bool = False,
    useColorDeconvolution: bool = False,
  ) -> None:
    # Store the root data directory.
    self.dataDir = Path(dataDir)
    # Store the batch size.
    self.batchSize = batchSize
    # Store the target image size.
    self.imageSize = imageSize
    # Store the number of workers.
    self.numWorkers = numWorkers
    # Store the augmentation flag.
    self.useAugmentation = useAugmentation
    # Store stain jitter flag.
    self.useStainJitter = useStainJitter
    # Store background removal flag.
    self.useBackgroundRemoval = useBackgroundRemoval
    # Store color deconvolution flag.
    self.useColorDeconvolution = useColorDeconvolution
    # Initialize the dictionary for dataset lengths.
    self.lengths = {}
    # Initialize the list for class names.
    self.classNames = []
    # Prepare the directory splits.
    self.rootPath = self._prepareSplits()
    # Build the dataset index to discover classes and calculate lengths.
    self._buildIndex()
    # Define the training transforms.
    self.trainTransform = self._getTrainTransform()
    # Define the validation and test transforms.
    self.valTestTransform = self._getValTestTransform()
    # Build the training pipeline.
    self.train = self._buildPipeline("train", isTraining=True)
    # Build the validation pipeline.
    self.val = self._buildPipeline("val", isTraining=False)
    # Build the test pipeline.
    self.test = self._buildPipeline("test", isTraining=False)

  # Check for train/val/test folders and auto-split if necessary.
  def _prepareSplits(self) -> Path:
    # Define the root path.
    rootPath = self.dataDir
    # Check if the train directory exists.
    if ((rootPath / "train").exists()):
      # Return the original root path.
      return rootPath
    # Check if there are class directories in the root path to split.
    classDirs = [
      d for d in rootPath.iterdir()
      if (d.is_dir() and d.name not in ["train", "val", "test", "SplitDataset"])
    ]
    # Check if class directories were found.
    if (len(classDirs) > 0):
      fprint("Train directory not found. Using split-folders to create train/val/test subsets...")
      # Import the splitfolders module dynamically.
      import splitfolders
      # Define the output directory for the split dataset.
      rootParent = rootPath.parent
      # Define the split output directory path.
      splitOutputDir = rootParent / "SplitDataset"
      # Check if the split directory does not exist.
      if (not splitOutputDir.exists()):
        # Use splitfolders to create the train/val/test subsets.
        splitfolders.ratio(input=str(rootPath), output=str(splitOutputDir), seed=1337, ratio=(0.8, 0.1, 0.1))
      # Return the new split directory path.
      return splitOutputDir
    else:
      # Raise an error if train directory is missing and no class directories are found.
      raise FileNotFoundError("Train directory not found and no class directories found to split.")

  # Scan the directories to discover class names and calculate split lengths.
  def _buildIndex(self) -> None:
    # Determine the target train directory.
    targetDir = self.rootPath / "train"
    # Get sorted list of class directories.
    classDirs = sorted([d for d in targetDir.iterdir() if (d.is_dir())])
    # Store the class names.
    self.classNames = [d.name for d in classDirs]
    fprint(f"Found {len(classDirs)} classes: {self.classNames}")
    # Iterate through train, val, and test splits to calculate lengths.
    for split in ["train", "val", "test"]:
      # Determine the split directory.
      splitDir = self.rootPath / split
      # Check if the split directory exists.
      if (splitDir.exists()):
        # Initialize the count for the split.
        count = 0
        # Iterate through class directories in the split.
        for classDir in classDirs:
          # Determine the class directory in the split.
          splitClassDir = splitDir / classDir.name
          # Check if the class directory exists in the split.
          if (splitClassDir.exists()):
            # Count the valid image files.
            for imgPath in splitClassDir.rglob("*"):
              # Filter by common image extensions.
              if (imgPath.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]):
                # Increment the count.
                count += 1
        # Store the length for the split.
        self.lengths[split.capitalize()] = count
        # Print the number of samples for the split.
        fprint(f"{split.capitalize()} samples: {count}")

  # Define the training transforms with data augmentation.
  def _getTrainTransform(self) -> transforms.Compose:
    if (not self.useAugmentation):
      # If augmentation is disabled, return only resizing and ToTensor.
      return transforms.Compose([
        transforms.Resize((self.imageSize, self.imageSize)),
        transforms.ToTensor()
      ])
    # Return the composed training transforms.
    transList = [
      # Resize the image slightly larger for random crop.
      transforms.Resize((int(self.imageSize * 1.1), int(self.imageSize * 1.1))),
      # Apply random resized cropping.
      transforms.RandomCrop(self.imageSize),
      # Apply random horizontal flipping.
      transforms.RandomHorizontalFlip(),
      # Apply random vertical flipping.
      transforms.RandomVerticalFlip(),
      # Apply random rotation for histopathology invariance.
      transforms.RandomRotation(360),
    ]
    if (self.useStainJitter):
      # Apply stain jittering if enabled.
      transList.append(StainJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    # Convert to Tensor and normalize to [0, 1].
    transList.append(transforms.ToTensor())
    return transforms.Compose(transList)

  # Define the validation and test transforms without augmentation.
  def _getValTestTransform(self) -> transforms.Compose:
    # Return the composed validation and test transforms.
    return transforms.Compose([
      # Resize the image to the target size.
      transforms.Resize((self.imageSize, self.imageSize)),
      # Convert to Tensor.
      transforms.ToTensor()
    ])

  # Build the PyTorch DataLoader pipeline for a specific split.
  def _buildPipeline(
    self,
    split: str,
    isTraining: bool,
  ) -> Optional[DataLoader]:
    # Determine the target split directory.
    splitDir = self.rootPath / split
    # Check if the split directory exists.
    if (not splitDir.exists()):
      # Return None for the missing split.
      return None
    # Select the appropriate transform based on the training flag.
    transform = self.trainTransform if (isTraining) else self.valTestTransform
    # Create the dataset object.
    dataset = GenericImageDataset(
      rootDir=self.rootPath,
      split=split,
      transform=transform,
      imageSize=self.imageSize,
      useBackgroundRemoval=self.useBackgroundRemoval,
      useColorDeconvolution=self.useColorDeconvolution,
    )

    # Initialize sampler.
    sampler = None
    # Check if training and weights exist.
    if (isTraining and hasattr(dataset, "weights") and len(dataset.weights) > 0):
      # Use WeightedRandomSampler for imbalanced histopathology data.
      sampler = WeightedRandomSampler(
        weights=dataset.weights,
        num_samples=len(dataset.weights),
        replacement=True
      )

    # Create and return the DataLoader.
    return DataLoader(
      dataset,
      batch_size=self.batchSize,
      shuffle=(sampler is None and isTraining),
      sampler=sampler,
      num_workers=self.numWorkers,
      pin_memory=True,
      persistent_workers=True if (self.numWorkers > 0) else False,
    )


class SoftSplit(torch.nn.Module):
  # Initialize the soft split module.
  def __init__(self, inCh=3, patchSize=7, stride=4, projDim=64):
    # Call the parent initialization.
    super().__init__()
    # Define the convolutional projection layer.
    self.proj = torch.nn.Conv2d(inCh, projDim, kernel_size=patchSize, stride=stride, padding=patchSize // 2)

  # Define the forward pass method.
  def forward(self, x):
    # Apply the projection layer.
    x = self.proj(x)
    # Get the batch size, channels, height, and width.
    b, c, h, w = x.shape
    # Flatten and transpose the tensor.
    return x.flatten(2).transpose(1, 2)


class TokenTransformerBlock(torch.nn.Module):
  # Initialize the transformer block.
  def __init__(self, dim, numHeads=4, mlpRatio=2.0):
    # Call the parent initialization.
    super().__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the multi-layer perceptron.
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(dim, int(dim * mlpRatio)),
      torch.nn.GELU(),
      torch.nn.Linear(int(dim * mlpRatio), dim)
    )

  # Define the forward pass.
  def forward(self, x):
    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection.
    x = res + attnOut
    # Add the MLP output.
    x = x + self.mlp(self.norm2(x))
    # Return the updated tensor.
    return x


class T2TModule(torch.nn.Module):
  # Initialize the T2T module.
  def __init__(self, inCh=3, tokenDim=64):
    # Call the parent initialization.
    super().__init__()
    # Define the first soft split.
    self.softSplit1 = SoftSplit(inCh, patchSize=7, stride=4, projDim=32)
    # Define the first transformer block.
    self.trans1 = TokenTransformerBlock(dim=32)
    # Define the second soft split.
    self.softSplit2 = SoftSplit(32, patchSize=3, stride=2, projDim=tokenDim)

  # Define the forward pass.
  def forward(self, x):
    # Apply the first soft split.
    x = self.softSplit1(x)
    # Apply the first transformer block.
    x = self.trans1(x)
    # Get the tensor shape.
    b, n, c = x.shape
    # Calculate the spatial dimensions.
    h = w = int(numpy.sqrt(n))
    # Reshape the tensor back to spatial format.
    x = x.transpose(1, 2).reshape(b, c, h, w)
    # Apply the second soft split.
    x = self.softSplit2(x)
    # Return the tokens.
    return x


class T2TViT(torch.nn.Module):
  # Initialize the T2T Vision Transformer.
  def __init__(self, numClasses=2, tokenDim=64, depth=4):
    # Call the parent initialization.
    super().__init__()
    # Define the T2T module.
    self.t2t = T2TModule(inCh=3, tokenDim=tokenDim)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.zeros(1, 1, tokenDim))
    # Define the positional embedding parameter.
    self.posEmbed = torch.nn.Parameter(torch.zeros(1, 1000, tokenDim))
    # Define the transformer blocks.
    self.blocks = torch.nn.ModuleList([TokenTransformerBlock(dim=tokenDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(tokenDim)
    # Define the classification head.
    self.head = torch.nn.Linear(tokenDim, numClasses)
    # Initialize the positional embedding.
    torch.nn.init.trunc_normal_(self.posEmbed, std=0.02)
    # Initialize the class token.
    torch.nn.init.trunc_normal_(self.clsToken, std=0.02)

  # Define the forward pass.
  def forward(self, x):
    # Get the batch size.
    b = x.shape[0]
    # Apply the T2T module.
    x = self.t2t(x)
    # Expand the class token.
    clsTokens = self.clsToken.expand(b, -1, -1)
    # Concatenate the class token with the patches.
    x = torch.cat((clsTokens, x), dim=1)
    # Check if positional embedding needs resizing.
    if (x.size(1) > self.posEmbed.size(1)):
      # Create a new positional embedding.
      newPos = torch.zeros(1, x.size(1), x.size(2), device=x.device)
      # Initialize the new positional embedding.
      torch.nn.init.trunc_normal_(newPos, std=0.02)
      # Update the positional embedding parameter.
      self.posEmbed = torch.nn.Parameter(newPos)
    # Add the positional embedding.
    x = x + self.posEmbed[:, :x.size(1), :]
    # Iterate through the transformer blocks.
    for block in self.blocks:
      # Apply the transformer block.
      x = block(x)
    # Apply the final normalization.
    x = self.norm(x)
    # Return the classification output.
    return self.head(x[:, 0])


class HierarchicalBlock(torch.nn.Module):
  # Initialize the hierarchical block.
  def __init__(self, dim, numHeads, mlpRatio=4.0):
    # Call the parent initialization.
    super().__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(dim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(dim, numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(dim)
    # Define the multi-layer perceptron.
    self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, int(dim * mlpRatio)), torch.nn.GELU(),
                                   torch.nn.Linear(int(dim * mlpRatio), dim))

  # Define the forward pass.
  def forward(self, x):
    # Store the residual connection.
    res = x
    # Apply the first normalization.
    x = self.norm1(x)
    # Compute the attention output.
    attnOut, _ = self.attn(x, x, x)
    # Add the residual connection.
    x = res + attnOut
    # Add the MLP output.
    x = x + self.mlp(self.norm2(x))
    # Return the updated tensor.
    return x


class PatchMerging(torch.nn.Module):
  # Initialize the patch merging module.
  def __init__(self, inDim, outDim):
    # Call the parent initialization.
    super().__init__()
    # Define the reduction linear layer.
    self.reduction = torch.nn.Linear(inDim * 4, outDim)
    # Define the layer normalization.
    self.norm = torch.nn.LayerNorm(inDim * 4)

  # Define the forward pass.
  def forward(self, x, h, w):
    # Get the tensor shape.
    b, l, c = x.shape
    # Reshape the tensor to spatial dimensions.
    x = x.view(b, h, w, c)
    # Check if padding is needed.
    if ((h % 2 == 1) or (w % 2 == 1)):
      # Pad the tensor.
      x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
    # Extract the four sub-patches.
    x0 = x[:, 0::2, 0::2, :]
    # Extract the second sub-patch.
    x1 = x[:, 1::2, 0::2, :]
    # Extract the third sub-patch.
    x2 = x[:, 0::2, 1::2, :]
    # Extract the fourth sub-patch.
    x3 = x[:, 1::2, 1::2, :]
    # Concatenate the sub-patches.
    x = torch.cat([x0, x1, x2, x3], -1)
    # Reshape the tensor.
    x = x.view(b, -1, 4 * c)
    # Apply the layer normalization.
    x = self.norm(x)
    # Apply the reduction layer.
    x = self.reduction(x)
    # Return the merged tensor and new dimensions.
    return x, (h + 1) // 2, (w + 1) // 2


class HierarchicalViT(torch.nn.Module):
  # Initialize the hierarchical Vision Transformer.
  def __init__(self, numClasses=2, embedDim=64):
    # Call the parent initialization.
    super().__init__()
    # Define the patch embedding layer.
    self.patchEmbed = torch.nn.Conv2d(3, embedDim, kernel_size=4, stride=4)
    # Define the first stage blocks.
    self.stage1 = torch.nn.ModuleList([HierarchicalBlock(embedDim, 4) for _ in range(2)])
    # Define the first patch merging layer.
    self.merge1 = PatchMerging(embedDim, embedDim * 2)
    # Define the second stage blocks.
    self.stage2 = torch.nn.ModuleList([HierarchicalBlock(embedDim * 2, 8) for _ in range(2)])
    # Define the second patch merging layer.
    self.merge2 = PatchMerging(embedDim * 2, embedDim * 4)
    # Define the third stage blocks.
    self.stage3 = torch.nn.ModuleList([HierarchicalBlock(embedDim * 4, 16) for _ in range(2)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim * 4)
    # Define the classification head.
    self.head = torch.nn.Linear(embedDim * 4, numClasses)

  # Define the forward pass.
  def forward(self, x):
    # Apply the patch embedding.
    x = self.patchEmbed(x)
    # Get the tensor shape.
    b, c, h, w = x.shape
    # Flatten and transpose the tensor.
    x = x.flatten(2).transpose(1, 2)
    # Iterate through the first stage blocks.
    for blk in self.stage1:
      # Apply the block.
      x = blk(x)
    # Apply the first patch merging.
    x, h, w = self.merge1(x, h, w)
    # Iterate through the second stage blocks.
    for blk in self.stage2:
      # Apply the block.
      x = blk(x)
    # Apply the second patch merging.
    x, h, w = self.merge2(x, h, w)
    # Iterate through the third stage blocks.
    for blk in self.stage3:
      # Apply the block.
      x = blk(x)
    # Apply the final normalization.
    x = self.norm(x)
    # Global average pooling.
    x = x.mean(dim=1)
    # Return the classification output.
    return self.head(x)


class PatchEmbedding(torch.nn.Module):
  # Initialize the patch embedding.
  def __init__(self, imgSize=128, patchSize=16, inChannels=3, embedDim=128):
    # Call the parent initialization.
    super().__init__()
    # Define the convolutional projection.
    self.proj = torch.nn.Conv2d(inChannels, embedDim, kernel_size=patchSize, stride=patchSize)

  # Define the forward pass.
  def forward(self, x):
    # Apply the projection and flatten.
    return self.proj(x).flatten(2).transpose(1, 2)


class TransformerEncoderBlock(torch.nn.Module):
  # Initialize the encoder block.
  def __init__(self, embedDim, numHeads, mlpDim):
    # Call the parent initialization.
    super().__init__()
    # Define the first layer normalization.
    self.norm1 = torch.nn.LayerNorm(embedDim)
    # Define the multi-head attention.
    self.attn = torch.nn.MultiheadAttention(embedDim, numHeads, batch_first=True)
    # Define the second layer normalization.
    self.norm2 = torch.nn.LayerNorm(embedDim)
    # Define the multi-layer perceptron.
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(embedDim, mlpDim),
      torch.nn.ReLU(),
      torch.nn.Linear(mlpDim, embedDim)
    )

  # Define the forward pass.
  def forward(self, x):
    # Compute the attention output.
    attnOutput, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
    # Add the residual connection.
    x = x + attnOutput
    # Add the MLP output.
    x = x + self.mlp(self.norm2(x))
    # Return the updated tensor.
    return x


class StandardViT(torch.nn.Module):
  # Initialize the standard Vision Transformer.
  def __init__(self, imgSize=128, patchSize=16, numClasses=2, embedDim=128, numHeads=4, depth=4, mlpDim=256):
    # Call the parent initialization.
    super().__init__()
    # Define the patch embedding.
    self.patchEmbedding = PatchEmbedding(imgSize, patchSize, 3, embedDim)
    # Define the class token parameter.
    self.clsToken = torch.nn.Parameter(torch.randn(1, 1, embedDim))
    # Define the positional encoding parameter.
    self.posEncoding = torch.nn.Parameter(torch.randn(1, (imgSize // patchSize) ** 2 + 1, embedDim))
    # Define the transformer blocks.
    self.transformerBlocks = torch.nn.ModuleList(
      [TransformerEncoderBlock(embedDim, numHeads, mlpDim) for _ in range(depth)])
    # Define the final layer normalization.
    self.norm = torch.nn.LayerNorm(embedDim)
    # Define the classification head.
    self.mlpHead = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass.
  def forward(self, x):
    # Get the batch size.
    b = x.size(0)
    # Apply the patch embedding.
    x = self.patchEmbedding(x)
    # Expand the class token.
    clsTokens = self.clsToken.expand(b, -1, -1)
    # Concatenate the class token.
    x = torch.cat((clsTokens, x), dim=1)
    # Add the positional encoding.
    x = x + self.posEncoding
    # Iterate through the transformer blocks.
    for block in self.transformerBlocks:
      # Apply the block.
      x = block(x)
    # Apply the final normalization.
    x = self.norm(x[:, 0])
    # Return the classification output.
    return self.mlpHead(x)


class CLIPClassifier(torch.nn.Module):
  # Initialize the CLIP classifier.
  def __init__(self, embedDim, numClasses):
    # Call the parent initialization.
    super().__init__()
    # Define the fully connected layer.
    self.fc = torch.nn.Linear(embedDim, numClasses)

  # Define the forward pass.
  def forward(self, x):
    # Return the classification output.
    return self.fc(x)


def BuildCLIPModel(numClasses, device):
  # Load the CLIP model and preprocessing.
  modelClip, preprocess = clip.load("ViT-B/16", device=device, jit=False)
  # Get the visual dimension.
  visualDim = modelClip.visual.output_dim
  # Initialize the classifier.
  classifier = CLIPClassifier(visualDim, numClasses).to(device)
  # Return the CLIP model and classifier.
  return modelClip, classifier


def BuildTimmModel(
  timmModelName: str,
  numClasses: int,
) -> torch.nn.Module:
  # Create the model using the timm library with pretrained weights.
  model = timm.create_model(timmModelName, pretrained=True, num_classes=numClasses)
  # Return the instantiated model.
  return model


# Define the function to build the ViT model.
def BuildViTModel(
    modelName: str,
    numClasses: int,
    device: str,
    imageSize: int = 224,
    usePretrainedCustomModels: bool = False,
    pretrainedModelName: str = "vit_base_patch16_224",
) -> Tuple[torch.nn.Module, Any]:
  # Check if the model is T2TViT.
  if (modelName == "T2TViT"):
    # Initialize the T2T Vision Transformer.
    model = T2TViT(numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is HierarchicalViT.
  elif (modelName == "HierarchicalViT"):
    # Initialize the Hierarchical Vision Transformer.
    model = HierarchicalViT(numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is StandardViT.
  elif (modelName == "StandardViT"):
    # Initialize the Standard Vision Transformer with the correct image size.
    model = StandardViT(imgSize=imageSize, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is CLIPViT.
  elif (modelName == "CLIPViT"):
    # Build the CLIP model and classifier.
    modelClip, classifier = BuildCLIPModel(numClasses, device)
    # Return the classifier and the CLIP model.
    return classifier, modelClip

  # Check if the model is SwinTransformerV2.
  elif (modelName == "SwinTransformerV2"):
    # SwinV2 requires 256x256 input. Override imageSize for this model.
    model = BuildTimmModel("hf-hub:timm/swinv2_base_window12to16_192to256_22kft1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is SwinTransformer.
  elif (modelName == "SwinTransformer"):
    # Build the Swin Transformer V1 model.
    model = BuildTimmModel("hf-hub:timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is DeiT.
  elif (modelName == "DeiT"):
    # Build the Data-efficient Image Transformer model using timm.
    model = BuildTimmModel("hf-hub:timm/deit_base_patch16_224.fb_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is ConvNeXtV2.
  elif (modelName == "ConvNeXtV2"):
    # Build the ConvNeXt V2 model with improved stability and performance.
    model = BuildTimmModel("hf-hub:timm/convnextv2_base.fcmae_ft_in22k_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is ConvNeXt.
  elif (modelName == "ConvNeXt"):
    # Build the ConvNeXt V1 hybrid CNN-Transformer model using timm.
    model = BuildTimmModel("hf-hub:timm/convnext_base.fb_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is MaxViT.
  elif (modelName == "MaxViT"):
    # Build the Multi-axis Vision Transformer model using timm.
    model = BuildTimmModel("hf-hub:timm/maxvit_tiny_rw_224.sw_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is BEiT.
  elif (modelName == "BEiT"):
    # Build the Bidirectional Encoder representation from Image Transformers model using timm.
    model = BuildTimmModel("hf-hub:timm/beit_base_patch16_224.in22k_ft_in22k_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is FastViT.
  elif (modelName == "FastViT"):
    # Build the FastViT model using timm.
    model = BuildTimmModel("hf-hub:timm/fastvit_t8.apple_in1k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is EVA02.
  elif (modelName == "EVA02"):
    # Build the EVA-02 model using timm.
    model = BuildTimmModel("hf-hub:timm/eva02_base_patch14_224.mim_in22k", numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is EVA02 Large (High Resolution)
  elif (modelName == "EVA02Large"):
    # Build the EVA-02 Large model pre-trained with MIM at 448x448 resolution.
    model = BuildTimmModel("hf-hub:timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", numClasses)
    # Note: You must set effectiveImageSize = 448 in CreateFitViTModel for this.
    return model, None

  # Check if the model is ConvNeXt Large CLIP
  elif (modelName == "ConvNeXtLargeCLIP"):
    # Build the ConvNeXt Large model with robust CLIP (LAION-2B) pre-training.
    model = BuildTimmModel("hf-hub:timm/convnext_large_mlp.clip_laion2b_augreg_ft_in1k", numClasses)
    return model, None

  # Check if the model is SwinTransformer Large 384
  elif (modelName == "SwinTransformerLarge384"):
    # Build the Swin Large model optimized for 384x384 high-resolution input.
    model = BuildTimmModel("hf-hub:timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k", numClasses)
    return model, None

  # Check if the model is EfficientNetV2 Large
  elif (modelName == "EfficientNetV2Large"):
    # Build the EfficientNetV2 Large model for highly efficient, robust feature extraction.
    model = BuildTimmModel("hf-hub:timm/efficientnetv2_rw_t.ra2_in1k", numClasses)
    return model, None

  # Check if the model is QuantumResNet.
  elif (modelName == "QuantumResNet"):
    # Build the Quantum ResNet hybrid model.
    model = BuildQuantumResNetModel(numClasses, device)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is VisionKANModel.
  elif (modelName == "VisionKANModel"):
    # Initialize the Vision KAN model.
    model = VisionKANModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is NeuralODEViTModel.
  elif (modelName == "NeuralODEViTModel"):
    # Initialize the Neural ODE ViT model.
    model = NeuralODEViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is SpikingViTModel.
  elif (modelName == "SpikingViTModel"):
    # Initialize the Spiking ViT model.
    model = SpikingViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None


  # Check if the model is HypernetworkViTModel.
  elif (modelName == "HypernetworkViTModel"):
    # Initialize the Hypernetwork ViT model.
    model = HypernetworkViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is LiquidSSMViTModel.
  elif (modelName == "LiquidSSMViTModel"):
    # Initialize the liquid state space vision transformer model.
    model = LiquidSSMViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is TestTimeEvolvingViTModel.
  elif (modelName == "TestTimeEvolvingViTModel"):
    # Initialize the test-time evolving vision transformer model.
    model = TestTimeEvolvingViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is TensorNetworkEntangledViTModel.
  elif (modelName == "TensorNetworkEntangledViTModel"):
    # Initialize the tensor network entangled vision transformer model.
    model = TensorNetworkEntangledViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is DiffusionPriorEnergyViTModel.
  elif (modelName == "DiffusionPriorEnergyViTModel"):
    # Initialize the diffusion prior energy vision transformer model.
    model = DiffusionPriorEnergyViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is FractalResonanceViTModel.
  elif (modelName == "FractalResonanceViTModel"):
    # Initialize the fractal resonance vision transformer model.
    model = FractalResonanceViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is TopologicalQuantumViTModel.
  elif (modelName == "TopologicalQuantumViTModel"):
    # Initialize the topological quantum vision transformer model.
    model = TopologicalQuantumViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is HolographicInterferenceViTModel.
  elif (modelName == "HolographicInterferenceViTModel"):
    # Initialize the holographic interference vision transformer model.
    model = HolographicInterferenceViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  # Check if the model is NeuromorphicLiquidStateViTModel.
  elif (modelName == "NeuromorphicLiquidStateViTModel"):
    # Initialize the neuromorphic liquid state vision transformer model.
    model = NeuromorphicLiquidStateViTModel(numClasses=numClasses)
    # Check if pretrained weights should be loaded.
    if (usePretrainedCustomModels):
      # Load pretrained ImageNet weights into the compatible layers.
      model = LoadPretrainedViTWeights(model, pretrainedModelName=pretrainedModelName, numClasses=numClasses)
    # Return the model and None for the secondary model.
    return model, None

  else:
    # Raise an error for unsupported models.
    raise ValueError(f"Unsupported ViT model: {modelName}")


# Define the TrainEpoch function.
def TrainEpoch(
  model: torch.nn.Module,
  dataLoader: DataLoader,
  criterion: torch.nn.Module,
  opt: torch.optim,
  device: str,
  numClasses: int,
  secondaryModel: Any = None,
  scaler: Any = None,
  accumulationSteps: int = 1,
  emaModel: Any = None,
  mixupFn: Any = None,
  useAmp: bool = False,
  topoWassersteinCriterion: Any = None,
) -> Tuple[float, float]:
  # Set the model to training mode.
  model.train()
  # Initialize the running loss.
  runningLoss = 0.0
  # Initialize the running correct predictions.
  runningCorrects = 0
  # Initialize the total samples.
  totalSamples = 0
  # Initialize the accumulation counter.
  accumulationCounter = 0
  # Iterate through the data loader.
  for inputs, labels, _ in dataLoader:
    # Move the inputs to the device.
    inputs = inputs.to(device)
    # Move the labels to the device.
    labels = labels.to(device)

    # Store hard labels for topological loss before any mixup modification.
    hardLabels = labels.clone() if (labels.dim() == 1) else torch.argmax(labels, dim=1)

    # Apply mixup or cutmix if provided.
    if (mixupFn is not None):
      # Safeguard: Skip mixup if the batch size is odd to prevent timm assertion errors.
      if (inputs.size(0) % 2 != 0):
        # Convert hard labels to one-hot soft targets for the loss function.
        labels = torch.nn.functional.one_hot(labels, numClasses).float()
      else:
        # Apply the mixup function to inputs and labels.
        inputs, labels = mixupFn(inputs, labels)

    # Check if using CLIP.
    if (secondaryModel is not None):
      # CLIP strictly requires exactly 3 channels. Adapt the input if necessary.
      if (inputs.shape[1] > 3):
        # Extract the first three channels for CLIP.
        clipInputs = inputs[:, :3, :, :]
      elif (inputs.shape[1] < 3):
        # Repeat the existing channel(s) to make it exactly 3 channels for CLIP.
        clipInputs = inputs.repeat(1, 3, 1, 1)[:, :3, :, :]
      else:
        # Use the input as is (already 3 channels).
        clipInputs = inputs
      # Extract features using the frozen CLIP visual encoder.
      with torch.no_grad():
        # Encode the images.
        clipFeatures = secondaryModel.encode_image(clipInputs).float()
      # Enable autocast if AMP is used.
      if (useAmp):
        # Use autocast for the forward pass.
        with torch.amp.autocast(device_type="cuda"):
          # Check if topological loss is enabled.
          if (topoWassersteinCriterion is not None):
            # Extract both logits and intermediate features.
            outputs, features = model(clipFeatures)
          else:
            # Unpack logits and discard features if topological loss is disabled.
            outputs, _ = model(clipFeatures)
      else:
        # Check if topological loss is enabled.
        if (topoWassersteinCriterion is not None):
          # Extract both logits and intermediate features.
          outputs, features = model(clipFeatures)
        else:
          # Unpack logits and discard features if topological loss is disabled.
          outputs, _ = model(clipFeatures)
    else:
      # Enable autocast if AMP is used.
      if (useAmp):
        # Use autocast for the forward pass.
        with torch.amp.autocast(device_type="cuda"):
          # Check if topological loss is enabled.
          if (topoWassersteinCriterion is not None):
            # Extract both logits and intermediate features.
            outputs, features = model(inputs)
          else:
            # Unpack logits and discard features if topological loss is disabled.
            outputs, _ = model(inputs)
      else:
        # Perform the forward pass without AMP.
        # Check if topological loss is enabled.
        if (topoWassersteinCriterion is not None):
          # Extract both logits and intermediate features.
          outputs, features = model(inputs)
        else:
          # Unpack logits and discard features if topological loss is disabled.
          outputs, _ = model(inputs)

    # Compute the base classification loss.
    baseLoss = criterion(outputs, labels)

    # Initialize the topological Wasserstein loss.
    topoLoss = torch.tensor(0.0, device=device)

    # Check if the topological Wasserstein criterion is provided.
    if (topoWassersteinCriterion is not None):
      # Generate a dummy spatial prior for the topological penalty calculation.
      spatialPrior = torch.ones(features.size(0), device=device) / features.size(0)
      # Compute the topological and contrastive loss components using hard labels.
      lossDict = topoWassersteinCriterion(features, hardLabels, spatialPrior)
      # Extract the total topological loss from the returned dictionary.
      topoLoss = lossDict["TotalLoss"]

    # Combine the base loss and the topological loss.
    loss = baseLoss + topoLoss

    # Capture the actual loss value for logging BEFORE AMP scaling modifies it.
    actualLoss = loss.item()

    # Normalize the loss by accumulation steps.
    loss = loss / accumulationSteps
    # Scale the loss if AMP is used.
    if (useAmp and scaler is not None):
      # Scale the loss for backpropagation.
      loss = scaler.scale(loss)
    # Perform the backward pass.
    loss.backward()
    # Increment the accumulation counter.
    accumulationCounter += 1
    # Check if it is time to update the weights.
    if (accumulationCounter % accumulationSteps == 0):
      # Update the weights using the scaler if AMP is used.
      if (useAmp and scaler is not None):
        # Step the optimizer (this internally unscales gradients and checks for infs/nans).
        scaler.step(opt)
        # Update the scaler's state for the next iteration.
        scaler.update()
      else:
        # Update the weights normally.
        opt.step()
      # Zero the gradients.
      opt.zero_grad()
      # Update the EMA model if provided.
      if (emaModel is not None):
        # Update the EMA parameters.
        emaModel.update_parameters(model)
    # Accumulate the ACTUAL loss (not the scaled one).
    runningLoss += actualLoss * inputs.size(0)

    # Get the predictions.
    if (mixupFn is not None):
      # Get the max indices for multiclass classification with soft labels.
      _, preds = torch.max(outputs, 1)
      # Get the true labels for accuracy calculation.
      if (labels.dim() > 1):
        # Get the argmax of the soft labels.
        trueLabels = torch.argmax(labels, dim=1)
      else:
        # Use the labels directly.
        trueLabels = labels
      # Accumulate the correct predictions.
      runningCorrects += torch.sum(preds == trueLabels.data)
      # Accumulate the total samples.
      totalSamples += inputs.size(0)
      # Continue to the next batch.
      continue
    # Get the max indices for classification (works for both 2 and >2 classes).
    _, preds = torch.max(outputs, 1)
    # Accumulate the correct predictions.
    runningCorrects += torch.sum(preds == labels.data)
    # Accumulate the total samples.
    totalSamples += inputs.size(0)
  # Compute the epoch loss.
  epochLoss = runningLoss / totalSamples
  # Compute the epoch accuracy.
  epochAcc = runningCorrects.double() / totalSamples
  # Return the epoch loss and accuracy.
  return epochLoss, epochAcc.item()


# Define the ValidateEpoch function.
def ValidateEpoch(
  model: torch.nn.Module,
  dataLoader: DataLoader,
  criterion: torch.nn.Module,
  device: str,
  numClasses: int,
  secondaryModel: Any = None,
  emaModel: Any = None,
  useAmp: bool = False,
) -> Tuple[float, float]:
  # Set the model to evaluation mode.
  model.eval()
  # Set the EMA model to evaluation mode if provided.
  if (emaModel is not None):
    # Evaluate the EMA model.
    emaModel.eval()
  # Initialize the running loss.
  runningLoss = 0.0
  # Initialize the running correct predictions.
  runningCorrects = 0
  # Initialize the total samples.
  totalSamples = 0
  # Disable gradient computation.
  with torch.no_grad():
    # Iterate through the data loader.
    for inputs, labels, _ in dataLoader:
      # Move the inputs to the device.
      inputs = inputs.to(device)
      # Move the labels to the device.
      labels = labels.to(device)
      # Select the model to use for validation.
      evalModel = emaModel if (emaModel is not None) else model
      # Check if using CLIP.
      if (secondaryModel is not None):
        # CLIP strictly requires 3 channels. Slice the input if necessary.
        if (inputs.shape[1] > 3):
          # Extract the first three channels for CLIP.
          clipInputs = inputs[:, :3, :, :]
        else:
          # Use the input as is.
          clipInputs = inputs
        # Extract features using the frozen CLIP visual encoder.
        features = secondaryModel.encode_image(clipInputs).float()
        # Enable autocast if AMP is used.
        if (useAmp):
          # Use autocast for the forward pass.
          with torch.amp.autocast(device_type="cuda"):
            # Pass features through the classifier.
            outputs_tuple = evalModel(features)
        else:
          # Pass features through the classifier without AMP.
          outputs_tuple = evalModel(features)
      else:
        # Enable autocast if AMP is used.
        if (useAmp):
          # Use autocast for the forward pass.
          with torch.amp.autocast(device_type="cuda"):
            # Perform the forward pass.
            outputs_tuple = evalModel(inputs)
        else:
          # Perform the forward pass without AMP.
          outputs_tuple = evalModel(inputs)

      # Unpack logits from the tuple returned by FeatureExtractorWrapper.
      if (isinstance(outputs_tuple, tuple)):
        outputs = outputs_tuple[0]
      else:
        outputs = outputs_tuple

      # Compute the loss.
      # Check if the criterion expects soft targets (e.g., when Mixup is used).
      if (isinstance(criterion, SoftTargetCrossEntropy)):
        # Convert hard labels to one-hot soft targets for validation.
        softLabels = torch.nn.functional.one_hot(labels, numClasses).float()
        # Compute the loss.
        loss = criterion(outputs, softLabels)
      else:
        # Compute the loss with hard targets.
        loss = criterion(outputs, labels)
      # Accumulate the loss.
      runningLoss += loss.item() * inputs.size(0)
      # Get the max indices for classification (works for both 2 and >2 classes).
      _, preds = torch.max(outputs, 1)
      # Accumulate the correct predictions.
      runningCorrects += torch.sum(preds == labels.data)
      # Accumulate the total samples.
      totalSamples += inputs.size(0)
  # Compute the epoch loss.
  epochLoss = runningLoss / totalSamples
  # Compute the epoch accuracy.
  epochAcc = runningCorrects.double() / totalSamples
  # Return the epoch loss and accuracy.
  return epochLoss, epochAcc.item()


# Define the function to adapt the first convolutional layer for arbitrary input channels.
def AdaptModelInputChannels(
  model: torch.nn.Module,
  targetChannels: int,
) -> torch.nn.Module:
  '''
  Finds the first Conv2d layer in the model and adapts its weights to accept "targetChannels".
  '''
  # Initialize variables to store the layer and its path.
  firstConv = None
  convPath = ""

  # Iterate through the model to find the first Conv2d layer.
  for name, module in model.named_modules():
    # Check if the current module is a Conv2d layer.
    if (isinstance(module, torch.nn.Conv2d)):
      # Store the reference to the layer.
      firstConv = module
      # Store the path to the layer.
      convPath = name
      # Break out of the loop since we only need the first one.
      break

  # Check if a Conv2d layer was found.
  if (firstConv is None):
    # Print a warning if no Conv2d layer is found.
    fprint("Warning: No Conv2d layer found to adapt input channels.")
    # Return the model unmodified.
    return model

  # Get the original number of input channels.
  originalChannels = firstConv.in_channels

  # Check if the channels already match.
  if (originalChannels == targetChannels):
    # Return the model unmodified.
    return model

  # Print the adaptation message.
  fprint(f"Adapting first Conv2d layer from {originalChannels} to {targetChannels} channels...")

  # Get the original weights.
  originalWeights = firstConv.weight.data

  # Check if we need to increase the number of channels.
  if (targetChannels > originalChannels):
    # Calculate the number of repeats needed.
    repeats = (targetChannels + originalChannels - 1) // originalChannels
    # Repeat the weights along the input channel dimension.
    newWeights = originalWeights.repeat(1, repeats, 1, 1)
    # Slice to the exact target channels.
    newWeights = newWeights[:, :targetChannels, :, :]
    # Normalize the weights to maintain the same magnitude of activations.
    newWeights = newWeights * (originalChannels / targetChannels)
  else:
    # Slice the weights to the target channels if reducing.
    newWeights = originalWeights[:, :targetChannels, :, :]

  # Create a new Conv2d layer with the updated input channels.
  newConv = torch.nn.Conv2d(
    in_channels=targetChannels,
    out_channels=firstConv.out_channels,
    kernel_size=firstConv.kernel_size,
    stride=firstConv.stride,
    padding=firstConv.padding,
    dilation=firstConv.dilation,
    groups=firstConv.groups,
    bias=(firstConv.bias is not None),
    padding_mode=firstConv.padding_mode
  )

  # Assign the new weights to the new layer.
  newConv.weight.data = newWeights

  # Check if the original layer had a bias.
  if (firstConv.bias is not None):
    # Copy the bias to the new layer.
    newConv.bias.data = firstConv.bias.data.clone()

  # Split the path to traverse the model hierarchy.
  pathParts = convPath.split(".")
  # Initialize the current module as the root model.
  currentModule = model

  # Traverse to the parent module.
  for part in pathParts[:-1]:
    # Check if the part is a digit (for Sequential or ModuleList).
    if (part.isdigit()):
      # Access by index.
      currentModule = currentModule[int(part)]
    else:
      # Access by attribute.
      currentModule = getattr(currentModule, part)

  # Get the final attribute name.
  finalAttr = pathParts[-1]

  # Replace the old layer with the new layer.
  setattr(currentModule, finalAttr, newConv)

  # Return the modified model.
  return model


# Define the function to create and fit the ViT model.
def CreateFitViTModel(
    trainLoader: DataLoader,
    valLoader: DataLoader,
    modelName: str,
    numClasses: int,
    numEpochs: int = 50,
    learningRate: float = 1e-4,
    device: str = "cuda",
    outputDir: str = "Results",
    patience: int = 10,
    imageSize: int = 224,
    optimizerName: str = "Adam",
    useAmp: bool = False,
    accumulationSteps: int = 1,
    useEma: bool = False,
    useMixup: bool = False,
    mixupAlpha: float = 0.2,
    cutmixAlpha: float = 1.0,
    lossFunction: str = "CrossEntropy",
    labelSmoothing: float = 0.0,
    schedulerName: str = "None",
    useTopoWassersteinLoss: bool = True,
    epsilon: float = 0.1,
    lambdaTopo: float = 0.1,
    lambdaContrastive: float = 0.1,
    usePretrainedCustomModels: bool = False,
    pretrainedModelName: str = "vit_base_patch16_224",
) -> Tuple[torch.nn.Module, Dict[str, List[float]], Any, Dict[str, Any]]:
  # Determine effective image size based on model.
  effectiveImageSize = imageSize
  if (modelName == "SwinTransformerV2"):
    effectiveImageSize = 256
  elif (modelName == "SwinTransformerLarge384"):
    effectiveImageSize = 384
  elif (modelName == "EVA02Large"):
    effectiveImageSize = 448

  # Build the ViT model with the specified image size.
  # secondaryModel is used for models like CLIP that require a separate visual encoder.
  # Build the ViT model with the specified image size.
  baseModel, secondaryModel = BuildViTModel(
    modelName=modelName,
    numClasses=numClasses,
    device=device,
    imageSize=effectiveImageSize,
    usePretrainedCustomModels=usePretrainedCustomModels,
    pretrainedModelName=pretrainedModelName,
  )

  # Wrap the base model to enable feature extraction for topological loss.
  model = FeatureExtractorWrapper(baseModel)

  # Determine the input channels from the dataset by peeking at one batch.
  sampleInputs, _, _ = next(iter(trainLoader))
  targetChannels = sampleInputs.shape[1]

  # Adapt the model's first layer to match the dataset's channel count.
  model = AdaptModelInputChannels(model, targetChannels)

  # Move the model to the specified device.
  model = model.to(device)

  # Initialize the mixup function if requested.
  mixupFn = None
  # Check if mixup or cutmix is enabled.
  if (useMixup):
    # Initialize the Mixup and Cutmix handler from timm.
    mixupFn = TimmMixup(
      mixup_alpha=mixupAlpha,
      cutmix_alpha=cutmixAlpha,
      prob=1.0,
      num_classes=numClasses,
      label_smoothing=labelSmoothing,
    )
  # Check if mixup or cutmix is enabled for loss selection.
  if (useMixup):
    # Use Soft Target Cross Entropy for mixup/cutmix.
    criterion = SoftTargetCrossEntropy()
  else:
    # Define the loss function based on the configuration.
    # Note: Since all models are built with `num_classes=N` (where N=2 for binary),
    # they output N logits. Therefore, we use CrossEntropy-based losses for both binary and multiclass.
    if (lossFunction == "Focal"):
      # Get class counts from training dataset.
      trainLabels = trainLoader.dataset.labels
      classCounts = numpy.bincount(trainLabels, minlength=numClasses).tolist()
      # Use Class-Balanced Focal Loss for imbalanced datasets.
      criterion = FocalLoss(numClasses=numClasses, classCounts=classCounts)
    elif (lossFunction == "LabelSmoothing"):
      # Use Label Smoothing Cross Entropy for both binary and multiclass.
      criterion = LabelSmoothingCrossEntropy(smoothing=labelSmoothing)
    else:
      # Use standard Cross Entropy Loss for both binary and multiclass.
      criterion = torch.nn.CrossEntropyLoss()

  # Initialize the Wasserstein Topological Loss module conditionally.
  topoWassersteinCriterion = None
  if (useTopoWassersteinLoss):
    topoWassersteinCriterion = WassersteinTopologicalLoss(
      numClasses=numClasses,
      featureDim=1,
      epsilon=epsilon,
      lambdaTopo=lambdaTopo,
      lambdaContrastive=lambdaContrastive
    ).to(device)

  # Define the model checkpoint path.
  modelCheckpointPath = str(Path(outputDir) / "BestModel.pt")
  # Initialize the best validation accuracy for early stopping.
  bestValAcc = 0.0
  # Initialize the best validation loss for early stopping.
  bestValLoss = float("inf")
  bestValAcc = float("-inf")
  # Initialize the counter for epochs without improvement.
  epochsNoImprove = 0
  # Initialize the training history dictionary.
  history = {"TrainLoss": [], "ValLoss": [], "TrainAcc": [], "ValAcc": []}
  # Check if using CLIP to freeze the visual encoder.
  if (secondaryModel is not None):
    # Freeze the CLIP visual encoder parameters.
    for param in secondaryModel.parameters():
      # Set requires_grad to False.
      param.requires_grad = False
  # Check if the optimizer is AdamW.
  if (optimizerName == "AdamW"):
    # Use the AdamW optimizer.
    opt = torch.optim.AdamW(model.parameters(), lr=learningRate)
  # Check if the optimizer is SGD.
  elif (optimizerName == "SGD"):
    # Use the SGD optimizer with momentum.
    opt = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=0.9)
  # Check if the optimizer is RMSprop.
  elif (optimizerName == "RMSprop"):
    # Use the RMSprop optimizer.
    opt = torch.optim.RMSprop(model.parameters(), lr=learningRate)
  # Check if the optimizer is RAdam.
  elif (optimizerName == "RAdam"):
    # Use the built-in Rectified Adam optimizer.
    opt = torch.optim.RAdam(model.parameters(), lr=learningRate)
  # Check if the optimizer is Lion.
  elif (optimizerName == "Lion"):
    # Use the Lion optimizer.
    opt = Lion(model.parameters(), lr=learningRate)
  # Check if the optimizer is Prodigy.
  elif (optimizerName == "Prodigy"):
    # Use the Prodigy optimizer with its default learning rate of 1.0.
    opt = Prodigy(model.parameters(), lr=1.0)
  # Check if the optimizer is ScheduleFreeAdamW.
  elif (optimizerName == "ScheduleFreeAdamW"):
    # Use the Schedule-Free AdamW optimizer.
    opt = AdamWScheduleFree(model.parameters(), lr=learningRate)
  # Check if the optimizer is Sophia.
  elif (optimizerName == "Sophia"):
    # Use the locally defined Sophia optimizer.
    opt = SophiaG(model.parameters(), lr=learningRate)
  else:
    # Default to the standard Adam optimizer.
    opt = torch.optim.Adam(model.parameters(), lr=learningRate)
  # Initialize the EMA model if requested.
  emaModel = None
  # Check if EMA is enabled.
  if (useEma):
    # Import the AveragedModel for EMA.
    from torch.optim.swa_utils import AveragedModel
    # Create the EMA model.
    emaModel = AveragedModel(model)
  # Initialize the learning rate scheduler.
  scheduler = None
  # Check the scheduler name.
  if (schedulerName == "CosineWarmup"):
    # Use Cosine Annealing with Linear Warmup.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=numEpochs, eta_min=1e-6)
  # Initialize the gradient scaler for AMP.
  scaler = torch.amp.GradScaler() if (useAmp) else None
  # Train for the specified number of epochs.
  for epoch in range(numEpochs):
    # Train for one epoch.
    trainLoss, trainAcc = TrainEpoch(
      model, trainLoader, criterion, opt, device, numClasses, secondaryModel,
      scaler, accumulationSteps, emaModel, mixupFn, useAmp, topoWassersteinCriterion
    )
    # Validate for one epoch.
    valLoss, valAcc = ValidateEpoch(
      model, valLoader, criterion, device, numClasses, secondaryModel, emaModel, useAmp
    )
    # Update the scheduler if provided.
    if (scheduler is not None):
      # Step the scheduler.
      scheduler.step()
    # Append the metrics to the history.
    history["TrainLoss"].append(trainLoss)
    # Append the validation loss to the history.
    history["ValLoss"].append(valLoss)
    # Append the training accuracy to the history.
    history["TrainAcc"].append(trainAcc)
    # Append the validation accuracy to the history.
    history["ValAcc"].append(valAcc)
    # Print the epoch metrics.
    fprint(
      f"Epoch {epoch + 1}/{numEpochs} - "
      f"Train Loss: {trainLoss:.4f} - Train Acc: {trainAcc:.4f} - "
      f"Val Loss: {valLoss:.4f} - Val Acc: {valAcc:.4f}"
    )

    # --- Early Stopping Logic Start ---
    # Check if the validation loss has improved (strictly less than)
    if (valLoss < bestValLoss or valAcc > bestValAcc):
      # Update the best validation loss.
      bestValLoss = valLoss
      # Update the best validation accuracy.
      bestValAcc = valAcc
      # Reset the epochs without improvement counter.
      epochsNoImprove = 0
      # Save the best model weights.
      if (emaModel is not None):
        # Save the underlying model state dict from EMA to avoid "module." prefix issues.
        torch.save(emaModel.module.state_dict(), modelCheckpointPath)
      else:
        # Save the standard model state dict.
        torch.save(model.state_dict(), modelCheckpointPath)
      # Print a message about saving the new best model.
      fprint(
        f"  Saved new best model with validation loss = {bestValLoss:.4f} "
        f"and validation accuracy = {bestValAcc:.4f}"
      )
    else:
      # Increment the epochs without improvement counter.
      epochsNoImprove += 1

    # Check the early stopping condition.
    if (epochsNoImprove >= patience):
      # Print the early stopping message.
      fprint(
        f"  Early stopping triggered after {epoch + 1} epochs. "
        f"Best Val Loss: {bestValLoss:.4f}, Best Val Acc: {bestValAcc:.4f}"
      )
      # Break out of the training loop.
      break
    # --- Early Stopping Logic End ---

  # Load the best model weights from the checkpoint.
  model.load_state_dict(torch.load(modelCheckpointPath))
  # Create the configs' dictionary.
  configs = {
    "ModelName"          : str(modelName),
    "NumClasses"         : int(numClasses),
    "NumEpochs"          : int(numEpochs),
    "LearningRate"       : float(learningRate),
    "OptimizerName"      : str(optimizerName),
    "ModelCheckpointPath": str(modelCheckpointPath),
  }
  # Return the trained model, history, secondary model, and configs.
  return model, history, secondaryModel, configs


# Define the PlotTrainingHistory function.
def PlotTrainingHistory(
  history: Dict[str, List[float]],
  outputDir: str = "Results",
) -> None:
  # Configure matplotlib for high-quality output.
  plt.style.use("seaborn-v0_8-darkgrid")
  # Create a figure with two subplots for loss and accuracy curves.
  fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  # Plot the training and validation loss curves on the first subplot.
  axes[0].plot(history["TrainLoss"], label="Train Loss", linewidth=2)
  # Plot the validation loss curve on the first subplot.
  axes[0].plot(history["ValLoss"], label="Val Loss", linewidth=2)
  # Set the x-axis label for the loss plot.
  axes[0].set_xlabel("Epoch", fontsize=12)
  # Set the y-axis label for the loss plot.
  axes[0].set_ylabel("Loss", fontsize=12)
  # Set the title for the loss plot.
  axes[0].set_title("Loss Over Training Epochs", fontsize=14, fontweight="bold")
  # Add the legend to the loss plot.
  axes[0].legend(fontsize=10)
  # Enable the grid for the loss plot.
  axes[0].grid(True, alpha=0.3)
  # Plot the training accuracy curve on the second subplot.
  axes[1].plot(history["TrainAcc"], label="Train Acc", linewidth=2)
  # Plot the validation accuracy curve on the second subplot.
  axes[1].plot(history["ValAcc"], label="Val Acc", linewidth=2)
  # Set the x-axis label for the accuracy plot.
  axes[1].set_xlabel("Epoch", fontsize=12)
  # Set the y-axis label for the accuracy plot.
  axes[1].set_ylabel("Accuracy", fontsize=12)
  # Set the title for the accuracy plot.
  axes[1].set_title("Accuracy Over Training Epochs", fontsize=14, fontweight="bold")
  # Add the legend to the accuracy plot.
  axes[1].legend(fontsize=10)
  # Enable the grid for the accuracy plot.
  axes[1].grid(True, alpha=0.3)
  # Adjust the layout to prevent label overlap.
  plt.tight_layout()
  # Save the figure to the specified output directory.
  plt.savefig(Path(outputDir) / "TrainingHistory.png", dpi=720, bbox_inches="tight")
  # Close the plot to free memory.
  plt.close()
  # Print a confirmation message for user feedback.
  fprint(f"Training history plots saved to {outputDir}/TrainingHistory.png")


# Define the NumpyEncoder class.
class NumpyEncoder(json.JSONEncoder):
  '''
  Custom JSON encoder to handle numpy arrays and scalar types.
  '''

  # Define the default serialization method.
  def default(self, obj):
    # Check if the object is a numpy array.
    if (isinstance(obj, numpy.ndarray)):
      # Convert the numpy array to a native Python list.
      return obj.tolist()
    # Check if the object is a numpy scalar (integer or float).
    if (isinstance(obj, (numpy.integer, numpy.floating))):
      # Convert the numpy scalar to a native Python int or float.
      return obj.item()
    # Fallback to the default encoder for other types.
    return super().default(obj)


# Define the EvaluateModel function.
def EvaluateModel(
  model: torch.nn.Module,
  dataLoader: DataLoader,
  outputDir: str = "Results",
  savePlots: bool = True,
  classNames: Optional[List[str]] = None,
  prefix: str = "Test",
  device: str = "cuda",
  numClasses: int = 3,
  secondaryModel: Any = None,
) -> Dict[str, Union[float, numpy.ndarray, str]]:
  # Print the evaluation start message.
  fprint(f"Starting evaluation: savePlots={savePlots} | outputDir={outputDir} | prefix={prefix}")
  # Create a folder for the evaluation results if it doesn't exist for the current prefix.
  evalOutputDir = Path(outputDir) / prefix
  # Create the directory if it doesn't exist.
  evalOutputDir.mkdir(parents=True, exist_ok=True)
  # Initialize the list to collect predictions.
  allPreds = []
  # Initialize the list to collect ground truth labels.
  allLabels = []
  # Initialize the list to collect filenames (for detailed analysis).
  allFiles = []
  # Initialize the list to collect predicted probabilities (for detailed analysis).
  allPredProbs = []
  # Set the model to evaluation mode.
  model.eval()
  # Disable gradient computation.
  with torch.no_grad():
    # Initialize the progress bar for evaluation.
    progressBar = tqdm(dataLoader, desc="Evaluating", leave=False)
    # Iterate through evaluation batches.
    for inputs, labels, files in progressBar:
      # Move the inputs to the device.
      inputs = inputs.to(device)
      # Check if using CLIP.
      if (secondaryModel is not None):
        # CLIP strictly requires 3 channels. Slice the input if necessary.
        if (inputs.shape[1] > 3):
          # Extract the first three channels for CLIP.
          clipInputs = inputs[:, :3, :, :]
        else:
          # Use the input as is.
          clipInputs = inputs
        # Extract features using the frozen CLIP visual encoder.
        features = secondaryModel.encode_image(clipInputs).float()
        # Perform the forward pass to obtain model predictions.
        outputs_tuple = model(features)
      else:
        # Perform the forward pass to obtain model predictions.
        outputs_tuple = model(inputs)

      # Unpack logits from the tuple returned by FeatureExtractorWrapper.
      if (isinstance(outputs_tuple, tuple)):
        outputs = outputs_tuple[0]
      else:
        outputs = outputs_tuple
      # Apply softmax for both binary and multiclass classification (since model outputs [B, numClasses] logits).
      probs = torch.softmax(outputs, dim=1).cpu().numpy()
      # Get the predicted class indices.
      preds = numpy.argmax(probs, axis=1)
      # Append the batch predictions to the accumulation list.
      allPreds.extend(preds)
      # Append the batch labels to the accumulation list.
      allLabels.extend(labels.numpy())
      # Append the batch filenames to the accumulation list.
      allFiles.extend(files)
      # Append the batch predicted probabilities to the accumulation list.
      allPredProbs.extend(probs)
  # Convert the accumulated predictions list to a numpy array.
  allPreds = numpy.array(allPreds)
  # Convert the accumulated labels list to a numpy array.
  allLabels = numpy.array(allLabels)
  # Convert predicted probabilities to numpy array.
  allPredProbs = numpy.array(allPredProbs)

  # Generate the confusion matrix for detailed error analysis.
  confMatrix = confusion_matrix(allLabels, allPreds)
  # Check if class names are provided.
  if (classNames is not None):
    # Use the provided class names.
    targetNames = classNames
  else:
    # Generate default class names based on unique labels.
    targetNames = [f"Class{i}" for i in range(len(numpy.unique(allLabels)))]

  # Calculate the performance metrics.
  pmMetrics = CalculatePerformanceMetrics(confMatrix, addWeightedAverage=True, addPerClass=True)

  # Calculate Cohen's Kappa for agreement beyond chance.
  kappa = cohen_kappa_score(allLabels, allPreds)
  # Add Kappa to metrics.
  pmMetrics["CohensKappa"] = kappa

  # Calculate AUC-ROC and AUPRC if there are more than 1 class.
  try:
    # For multiclass, use "ovr" (One-vs-Rest) or "ovo" (One-vs-One). "ovr" is standard.
    # average="macro" calculates the mean of labels.
    aucRoc = roc_auc_score(allLabels, allPredProbs, multi_class="ovr", average="macro")
    auprc = average_precision_score(allLabels, allPredProbs, average="macro")

    pmMetrics["AUC_ROC_Macro"] = aucRoc
    pmMetrics["AUPRC_Macro"] = auprc

    # Plot ROC Curves if requested.
    if (savePlots):
      PlotROCCurves(allLabels, allPredProbs, targetNames, evalOutputDir, f"{prefix}_ROC.png")

  except ValueError:
    # Handle cases where AUC cannot be computed (e.g., only one class present in batch).
    pmMetrics["AUC_ROC_Macro"] = "N/A"
    pmMetrics["AUPRC_Macro"] = "N/A"

  # Generate the detailed classification report with per-class metrics.
  classReport = classification_report(
    allLabels, allPreds, target_names=targetNames, zero_division=0
  )
  # Save the confusion matrix plot if visualization is requested.
  if (savePlots):
    # Call the function to plot the confusion matrix.
    PlotConfusionMatrix(
      confMatrix,
      classNames=targetNames,
      outputDir=evalOutputDir,
      fileName=f"{prefix}CM.png",
    )
  # Save the confusion matrix to a CSV file with class names only in the first row and first column.
  with open(Path(evalOutputDir) / f"{prefix}CM.csv", "w", newline="") as csvFile:
    # Create a CSV writer object.
    csvWriter = csv.writer(csvFile)
    # Write the first row with an empty top-left corner and the class names as column headers.
    csvWriter.writerow([""] + targetNames)
    # Iterate through the confusion matrix rows.
    for i, className in enumerate(targetNames):
      # Write the row with the class name in the first column (row header) followed by the matrix values.
      csvWriter.writerow([className] + confMatrix[i].tolist())
  # Save the classification report to a text file for reference.
  if (savePlots):
    # Open the classification report file in write mode.
    with open(Path(evalOutputDir) / f"{prefix}ClassificationReport.txt", "w") as f:
      # Write the classification report to the file.
      f.write(classReport)
  # Compile all metrics into a structured dictionary for return.
  metrics = {
    **pmMetrics,
    "ConfusionMatrix"     : confMatrix.tolist(),
    "ClassificationReport": classReport,
    "Predictions"         : allPreds.tolist(),
    "Labels"              : allLabels.tolist(),
  }
  # Store the metrics in a JSON file for downstream use.
  with open(Path(evalOutputDir) / f"{prefix}EvaluationMetrics.json", "w") as jsonFile:
    # Dump the metrics dictionary to the JSON file.
    json.dump(metrics, jsonFile, indent=4, cls=NumpyEncoder)
  # Store detailed predictions by appending to a CSV file for detailed analysis.
  with open(Path(evalOutputDir) / f"{prefix}DetailedPredictions.csv", "w", newline="") as csvFile:
    # Create a CSV writer object.
    csvWriter = csv.writer(csvFile)
    # Write the header row for the detailed predictions CSV.
    csvWriter.writerow(["Split", "Filename", "Actual", "Pred", "PredProb"])
    # Iterate through the predictions, labels, filenames, and predicted probabilities.
    for filename, actual, pred, predProb in zip(allFiles, allLabels, allPreds, allPredProbs):
      # Write a row for each prediction with the split, filename, actual label, predicted label, and predicted probabilities.
      csvWriter.writerow(
        [prefix, filename, actual, pred, predProb.tolist() if isinstance(predProb, numpy.ndarray) else predProb])


# Define the PlotROCCurves function.
def PlotROCCurves(
  yTrue: numpy.ndarray,
  yScore: numpy.ndarray,
  classNames: List[str],
  outputDir: Path,
  fileName: str,
) -> None:
  '''
  Plots ROC curves for each class.
  '''
  # Determine number of classes.
  nClasses = len(classNames)
  # Create figure.
  plt.figure(figsize=(10, 8))

  # Compute ROC curve and ROC area for each class.
  for i in range(nClasses):
    # Binarize the output for One-vs-Rest.
    yTrueBin = (yTrue == i).astype(int)
    # Get scores for this class.
    yScoreClass = yScore[:, i]

    try:
      # Compute ROC curve.
      fpr, tpr, _ = roc_curve(yTrueBin, yScoreClass)
      # Compute AUC.
      rocAuc = roc_auc_score(yTrueBin, yScoreClass)
      # Plot.
      plt.plot(fpr, tpr, lw=2, label=f"{classNames[i]} (AUC = {rocAuc:.2f})")
    except ValueError:
      # Skip if only one class is present in this binarized view.
      continue

  # Plot diagonal line.
  plt.plot([0, 1], [0, 1], "k--", lw=2)
  # Set limits.
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  # Labels.
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Receiver Operating Characteristic (ROC) Curve")
  plt.legend(loc="lower right")
  plt.grid(True, alpha=0.3)

  # Save.
  plt.tight_layout()
  plt.savefig(outputDir / fileName, dpi=720, bbox_inches="tight")
  plt.close()
  fprint(f"ROC curves saved to {outputDir / fileName}")


# Define the PlotConfusionMatrix function.
def PlotConfusionMatrix(
  confMatrix: numpy.ndarray,
  classNames: List[str],
  outputDir: str = "Results",
  fileName: str = "CM.png",
  normalize: bool = False,
) -> None:
  # Apply row-wise normalization if percentage display is requested.
  if (normalize):
    # Calculate the normalized confusion matrix.
    confMatrixDisplay = confMatrix.astype("float") / confMatrix.sum(axis=1)[:, numpy.newaxis]
    # Set the format string for percentages.
    fmt = ".2%"
    # Set the annotation label.
    annotLabel = "Proportion"
  else:
    # Use the raw confusion matrix.
    confMatrixDisplay = confMatrix
    # Set the format string for integers.
    fmt = "d"
    # Set the annotation label.
    annotLabel = "Count"
  # Create a matplotlib figure with appropriate size for readability.
  plt.figure(figsize=(8, 6))
  # Generate the heatmap with annotations using the seaborn library.
  seaborn.heatmap(
    confMatrixDisplay,
    annot=True,
    fmt=fmt,
    cmap="Blues",
    xticklabels=classNames,
    yticklabels=classNames,
    cbar_kws={"label": annotLabel},
  )
  # Configure the x-axis label for clarity.
  plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
  # Configure the y-axis label for clarity.
  plt.ylabel("True Label", fontsize=12, fontweight="bold")
  # Configure the plot title for clarity.
  plt.title("Confusion Matrix", fontsize=14, fontweight="bold", pad=20)
  # Adjust the layout to prevent overlap.
  plt.tight_layout()
  # Save the figure to the specified output directory.
  plt.savefig(Path(outputDir) / fileName, dpi=720, bbox_inches="tight")
  # Close the plot to free memory.
  plt.close()
  # Print a confirmation message for user feedback.
  fprint(f"Confusion matrix saved to {Path(outputDir) / fileName}")


# Define the PlotClassHistograms function.
def PlotClassHistograms(
  dataLoaders: Dict[str, DataLoader],
  classNames: List[str],
  outputDir: str = "Results",
) -> None:
  # Initialize the dictionary for class counts.
  classCounts = {}
  # Initialize the dictionary for total split counts.
  splitTotals = {}
  # Iterate through each split in the data loaders.
  for splitName, loader in dataLoaders.items():
    # Access the labels directly from the dataset to avoid loading images from disk.
    labelsArr = numpy.array(loader.dataset.labels)
    # Count occurrences of each class.
    counts = numpy.bincount(labelsArr, minlength=len(classNames))
    # Store the counts for the split.
    classCounts[splitName] = counts
    # Store the total count for the split.
    splitTotals[splitName] = len(labelsArr)
  # Determine the number of classes.
  numClasses = len(classNames)
  # Determine the number of splits.
  numSplits = len(classCounts)
  # Check if there are no classes.
  if (numClasses == 0):
    # Return early if no classes.
    return
  # Create a figure for the histograms.
  plt.figure(figsize=(max(6, numClasses * 0.8), 5))
  # Define the width of the bars.
  barWidth = 0.25
  # Define the positions for the bars.
  r1 = numpy.arange(numClasses)
  # Initialize the index for splits.
  splitIdx = 0
  # Initialize the list to store bar containers for labeling.
  barContainers = []
  # Iterate through each split.
  for splitName, counts in classCounts.items():
    # Calculate the position for the current split.
    rPos = [x + splitIdx * barWidth for x in r1]
    # Get the total count for the current split.
    totalCount = splitTotals.get(splitName, 0)
    # Create the label with the count.
    labelWithCount = f"{splitName} (N={totalCount})"
    # Plot the bar chart for the current split and store the container.
    bars = plt.bar(rPos, counts, width=barWidth, edgecolor="grey", label=labelWithCount)
    # Append the bars to the list.
    barContainers.append(bars)
    # Increment the split index.
    splitIdx += 1
  # Set the x-axis ticks to the center of the groups.
  plt.xticks(
    [r + barWidth for r in range(numClasses)],
    classNames,
    rotation=45,
    ha="right",
  )
  # Set the x-axis label.
  plt.xlabel("Class", fontsize=12, fontweight="bold")
  # Set the y-axis label.
  plt.ylabel("Count", fontsize=12, fontweight="bold")
  # Set the plot title.
  plt.title("Class Distribution Across Splits", fontsize=14, fontweight="bold")
  # Add the legend.
  plt.legend()
  # Add count labels above each bar.
  for bars in barContainers:
    # Iterate through each bar in the container.
    for bar in bars:
      # Get the height of the bar.
      yVal = bar.get_height()
      # Check if the value is greater than 0 to avoid clutter.
      if (yVal > 0):
        # Add the text label above the bar.
        plt.text(
          bar.get_x() + bar.get_width() / 2.0,
          yVal,
          int(yVal),
          ha="center",
          va="bottom",
          fontsize=8,
        )
  # Adjust the layout.
  plt.tight_layout()
  # Construct the full file path for the histogram plot.
  histogramFilePath = str(Path(outputDir) / "ClassHistograms.png")
  # Save the figure.
  plt.savefig(histogramFilePath, dpi=720, bbox_inches="tight")
  # Close the plot.
  plt.close()
  # Print the confirmation message.
  fprint(f"Class histograms saved to {histogramFilePath}")


# Define the CalculatePerformanceMetrics function.
def CalculatePerformanceMetrics(
  confMatrix: numpy.ndarray,
  eps: float = 1e-10,
  addWeightedAverage: bool = False,
  addPerClass: bool = False,
) -> Dict[str, Any]:
  # Convert the confusion matrix to a NumPy array for easier manipulation.
  confMatrix = numpy.array(confMatrix)
  # Get the number of classes from the shape of the confusion matrix.
  noOfClasses = confMatrix.shape[0]
  # Check if the confusion matrix is for binary classification or multiclass.
  if (noOfClasses > 2):
    # Calculate True Positives (TP) as the diagonal elements of the confusion matrix.
    truePositives = numpy.diag(confMatrix)
    # Calculate False Positives (FP) as the sum of each column minus the TP.
    falsePositives = numpy.sum(confMatrix, axis=0) - truePositives
    # Calculate False Negatives (FN) as the sum of each row minus the TP.
    falseNegatives = numpy.sum(confMatrix, axis=1) - truePositives
    # Calculate True Negatives (TN) as the total sum of the matrix minus TP, FP, and FN.
    trueNegatives = numpy.sum(confMatrix) - (truePositives + falsePositives + falseNegatives)
  else:
    # For binary classification, the confusion matrix is a 2x2 matrix.
    # Unravel the confusion matrix to get the TN, FP, FN, and TP.
    trueNegatives, falsePositives, falseNegatives, truePositives = confMatrix.ravel()

  # Add a small epsilon value to avoid division by zero in metric calculations.
  truePositives = truePositives + eps
  # Add a small epsilon value to FP.
  falsePositives = falsePositives + eps
  # Add a small epsilon value to FN.
  falseNegatives = falseNegatives + eps
  # Add a small epsilon value to TN.
  trueNegatives = trueNegatives + eps

  # Create a dictionary to hold the calculated performance metrics and the TP, FP, FN, TN vectors.
  metrics = {
    # Store the true positives.
    "TruePositives" : str(truePositives),
    # Store the false positives.
    "FalsePositives": str(falsePositives),
    # Store the false negatives.
    "FalseNegatives": str(falseNegatives),
    # Store the true negatives.
    "TrueNegatives" : str(trueNegatives),
  }

  # If requested, calculate per-class precision, recall, F1, accuracy, and specificity.
  if (addPerClass and noOfClasses > 2):
    # Calculate per-class precision.
    precision = truePositives / (truePositives + falsePositives)
    # Calculate per-class recall.
    recall = truePositives / (truePositives + falseNegatives)
    # Calculate per-class F1.
    f1 = 2.0 * precision * recall / (precision + recall)
    # Calculate per-class accuracy.
    accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
    # Calculate per-class specificity.
    specificity = trueNegatives / (trueNegatives + falsePositives)
    # Calculate per-class balanced accuracy.
    bac = 0.5 * (recall + specificity)
    # Calculate per-class Matthews Correlation Coefficient.
    mcc = (truePositives * trueNegatives - falsePositives * falseNegatives) / numpy.sqrt(
      (truePositives + falsePositives) * (truePositives + falseNegatives) * (trueNegatives + falsePositives) * (
        trueNegatives + falseNegatives))
    # Calculate per-class Youden's J statistic.
    youden = recall + specificity - 1
    # Calculate per-class Yule's Q.
    yule = (truePositives * trueNegatives - falsePositives * falseNegatives) / (
      truePositives * trueNegatives + falsePositives * falseNegatives)

    # Iterate through each class to add metrics to the dictionary.
    for i in range(len(precision)):
      # Update the metrics dictionary with per-class metrics.
      metrics.update({
        f"Class {i} Precision"  : precision[i],
        f"Class {i} Recall"     : recall[i],
        f"Class {i} F1"         : f1[i],
        f"Class {i} Accuracy"   : accuracy[i],
        f"Class {i} Specificity": specificity[i],
        f"Class {i} BAC"        : bac[i],
        f"Class {i} MCC"        : mcc[i],
        f"Class {i} Youden"     : youden[i],
        f"Class {i} Yule"       : yule[i],
        f"TP Class {i}"         : truePositives[i],
        f"FP Class {i}"         : falsePositives[i],
        f"FN Class {i}"         : falseNegatives[i],
        f"TN Class {i}"         : trueNegatives[i],
      })

  # Calculate macro-averaged precision.
  precision = numpy.mean(truePositives / (truePositives + falsePositives))
  # Calculate macro-averaged recall.
  recall = numpy.mean(truePositives / (truePositives + falseNegatives))
  # Calculate macro-averaged F1.
  f1 = 2.0 * precision * recall / (precision + recall)
  # Calculate macro-averaged accuracy.
  accuracy = numpy.mean(truePositives + trueNegatives) / numpy.sum(confMatrix)
  # Calculate macro-averaged specificity.
  specificity = numpy.mean(trueNegatives / (trueNegatives + falsePositives))
  # Calculate macro-averaged balanced accuracy.
  bac = 0.5 * (recall + specificity)
  # Calculate macro-averaged Matthews Correlation Coefficient.
  mcc = numpy.mean((truePositives * trueNegatives - falsePositives * falseNegatives) / numpy.sqrt(
    (truePositives + falsePositives) * (truePositives + falseNegatives) * (trueNegatives + falsePositives) * (
      trueNegatives + falseNegatives)))
  # Calculate macro-averaged Youden's J statistic.
  youden = recall + specificity - 1
  # Calculate macro-averaged Yule's Q.
  yule = numpy.mean((truePositives * trueNegatives - falsePositives * falseNegatives) / (
    truePositives * trueNegatives + falsePositives * falseNegatives))

  # Add macro metrics to the dictionary.
  metrics.update({
    "Macro Precision"  : precision,
    "Macro Recall"     : recall,
    "Macro F1"         : f1,
    "Macro Accuracy"   : accuracy,
    "Macro Specificity": specificity,
    "Macro BAC"        : bac,
    "Macro MCC"        : mcc,
    "Macro Youden"     : youden,
    "Macro Yule"       : yule,
  })

  # If requested, calculate the macro average of the metrics.
  if (addWeightedAverage):
    # Calculate the average of the 6 main metrics.
    avg = (precision + recall + f1 + accuracy + specificity + bac) / 6.0
    # Calculate the average of all 9 metrics.
    avg9 = (precision + recall + f1 + accuracy + specificity + bac + mcc + youden + yule) / 9.0
    # Update the metrics dictionary with the averages.
    metrics.update({
      "Macro Average"  : avg,
      "Macro Average 9": avg9,
    })

  # Calculate micro-averaged precision.
  precision = numpy.sum(truePositives) / numpy.sum(truePositives + falsePositives)
  # Calculate micro-averaged recall.
  recall = numpy.sum(truePositives) / numpy.sum(truePositives + falseNegatives)
  # Calculate micro-averaged F1.
  f1 = 2.0 * precision * recall / (precision + recall)
  # Calculate micro-averaged accuracy.
  accuracy = numpy.sum(truePositives + trueNegatives) / numpy.sum(
    truePositives + trueNegatives + falsePositives + falseNegatives)
  # Calculate micro-averaged specificity.
  specificity = numpy.sum(trueNegatives) / numpy.sum(trueNegatives + falsePositives)
  # Calculate micro-averaged balanced accuracy.
  bac = 0.5 * (recall + specificity)
  # Calculate micro-averaged Matthews Correlation Coefficient.
  mcc = (
    (numpy.sum(truePositives) * numpy.sum(trueNegatives) - numpy.sum(falsePositives) * numpy.sum(falseNegatives)) /
    numpy.sqrt(
      (numpy.sum(truePositives) + numpy.sum(falsePositives)) * (
        numpy.sum(truePositives) + numpy.sum(falseNegatives)) * (
        numpy.sum(trueNegatives) + numpy.sum(falsePositives)) * (numpy.sum(trueNegatives) + numpy.sum(falseNegatives))
    )
  )
  # Calculate micro-averaged Youden's J statistic.
  youden = recall + specificity - 1
  # Calculate micro-averaged Yule's Q.
  yule = (numpy.sum(truePositives) * numpy.sum(trueNegatives) - numpy.sum(falsePositives) * numpy.sum(
    falseNegatives)) / (numpy.sum(truePositives) * numpy.sum(trueNegatives) + numpy.sum(falsePositives) * numpy.sum(
    falseNegatives))

  # Add micro metrics to the dictionary.
  metrics.update({
    "Micro Precision"  : precision,
    "Micro Recall"     : recall,
    "Micro F1"         : f1,
    "Micro Accuracy"   : accuracy,
    "Micro Specificity": specificity,
    "Micro BAC"        : bac,
    "Micro MCC"        : mcc,
    "Micro Youden"     : youden,
    "Micro Yule"       : yule,
  })

  # If requested, calculate the micro average of the metrics.
  if (addWeightedAverage):
    # Calculate the average of the 6 main metrics.
    avg = (precision + recall + f1 + accuracy + specificity + bac) / 6.0
    # Calculate the average of all 9 metrics.
    avg9 = (precision + recall + f1 + accuracy + specificity + bac + mcc + youden + yule) / 9.0
    # Update the metrics dictionary with the averages.
    metrics.update({
      "Micro Average"  : avg,
      "Micro Average 9": avg9,
    })

  # Calculate the number of samples per class by summing the rows of the confusion matrix.
  samples = numpy.sum(confMatrix, axis=1)
  # Calculate the weights for each class as the proportion of samples in that class.
  weights = samples / numpy.sum(confMatrix)

  # Calculate weighted-averaged precision.
  precision = numpy.sum(truePositives / (truePositives + falsePositives) * weights)
  # Calculate weighted-averaged recall.
  recall = numpy.sum(truePositives / (truePositives + falseNegatives) * weights)
  # Calculate weighted-averaged F1.
  f1 = 2.0 * precision * recall / (precision + recall)
  # Calculate weighted-averaged accuracy.
  accuracy = numpy.sum((truePositives + trueNegatives) * weights) / numpy.sum(confMatrix)
  # Calculate weighted-averaged specificity.
  specificity = numpy.sum(trueNegatives / (trueNegatives + falsePositives) * weights)
  # Calculate weighted-averaged balanced accuracy.
  bac = 0.5 * (recall + specificity)
  # Calculate weighted-averaged Matthews Correlation Coefficient.
  mcc = numpy.sum((truePositives * trueNegatives - falsePositives * falseNegatives) / numpy.sqrt(
    (truePositives + falsePositives) * (truePositives + falseNegatives) * (trueNegatives + falsePositives) * (
      trueNegatives + falseNegatives)) * weights)
  # Calculate weighted-averaged Youden's J statistic.
  youden = recall + specificity - 1
  # Calculate weighted-averaged Yule's Q.
  yule = numpy.sum((truePositives * trueNegatives - falsePositives * falseNegatives) / (
    truePositives * trueNegatives + falsePositives * falseNegatives) * weights)

  # Add weights and weighted metrics to the dictionary.
  metrics.update({
    "Weights"             : weights,
    "Weighted Precision"  : precision,
    "Weighted Recall"     : recall,
    "Weighted F1"         : f1,
    "Weighted Accuracy"   : accuracy,
    "Weighted Specificity": specificity,
    "Weighted BAC"        : bac,
    "Weighted MCC"        : mcc,
    "Weighted Youden"     : youden,
    "Weighted Yule"       : yule,
  })

  # If requested, calculate the weighted average of the metrics.
  if (addWeightedAverage):
    # Calculate the average of the 6 main metrics.
    avg = (precision + recall + f1 + accuracy + specificity + bac) / 6.0
    # Calculate the average of all 9 metrics.
    avg9 = (precision + recall + f1 + accuracy + specificity + bac + mcc + youden + yule) / 9.0
    # Update the metrics dictionary with the averages.
    metrics.update({
      "Weighted Average"  : avg,
      "Weighted Average 9": avg9,
    })

  # Return the dictionary containing all calculated metrics.
  return metrics


# Define the function to run the complete pipeline.
def RunCompletePipeline(
    dataDir: str,
    outputDir: str = "Results",
    modelName: str = "StandardViT",
    numEpochs: int = 50,
    batchSize: int = 16,
    imageSize: int = 224,
    learningRate: float = 1e-4,
    devicePref: Optional[str] = "auto",
    patience: int = 10,
    optimizerName: str = "Adam",
    useAmp: bool = False,
    accumulationSteps: int = 1,
    useEma: bool = False,
    useMixup: bool = False,
    mixupAlpha: float = 0.2,
    cutmixAlpha: float = 1.0,
    lossFunction: str = "CrossEntropy",
    labelSmoothing: float = 0.0,
    schedulerName: str = "None",
    useAugmentation: bool = True,
    useStainJitter: bool = False,
    useBackgroundRemoval: bool = False,
    useColorDeconvolution: bool = False,
    useTopoWassersteinLoss: bool = True,
    epsilon: float = 0.1,
    lambdaTopo: float = 0.1,
    lambdaContrastive: float = 0.1,
    usePretrainedCustomModels: bool = False,
    pretrainedModelName: str = "vit_base_patch16_224",
) -> Dict[str, Any]:
  # Print the pipeline initialization message.
  fprint("Starting ViT Image Classification Pipeline")
  # Determine the computation device.
  if (devicePref == "auto"):
    # Use CUDA if available, otherwise CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
  else:
    # Use the specified device.
    device = devicePref
  # Print the selected device.
  fprint(f"Using device: {device}")
  # Print the data directory.
  fprint(f"Data Directory: {dataDir}")
  # Print the output directory.
  fprint(f"Output Directory: {outputDir}")
  # Create the output directory structure for organized results.
  Path(outputDir).mkdir(parents=True, exist_ok=True)

  # Determine effective image size based on model.
  effectiveImageSize = imageSize
  if (modelName == "SwinTransformerV2"):
    effectiveImageSize = 256
    fprint(f"Detected SwinTransformerV2. Overriding image size to {effectiveImageSize}.")

  # Generate data loaders using the unified pipeline.
  fprint("Preparing data loaders...")
  # Instantiate the data pipeline with the EFFECTIVE image size.
  pipeline = PyTorchFolderBasedDataPipeline(
    dataDir=dataDir,
    batchSize=batchSize,
    imageSize=effectiveImageSize,
    useAugmentation=useAugmentation,
    useStainJitter=useStainJitter,
    useBackgroundRemoval=useBackgroundRemoval,
    useColorDeconvolution=useColorDeconvolution,
  )
  # Print the data loaders ready message.
  fprint("Data loaders ready.")
  # Collect available loaders into a dictionary.
  availableLoaders = {}
  # Check if the training loader exists.
  if (pipeline.train):
    # Add the training loader to the dictionary.
    availableLoaders["Train"] = pipeline.train
  # Check if the validation loader exists.
  if (pipeline.val):
    # Add the validation loader to the dictionary.
    availableLoaders["Val"] = pipeline.val
  # Handle the case where the validation loader is missing.
  else:
    # Print a warning about the missing validation loader.
    fprint("Warning: Validation loader not found. Using testing loader for validation.")
    # Assign the test loader to the validation loader variable.
    pipeline.val = pipeline.test
    # Add the assigned loader to the dictionary.
    availableLoaders["Val"] = pipeline.val
  # Check if the test loader exists.
  if (pipeline.test):
    # Add the test loader to the dictionary.
    availableLoaders["Test"] = pipeline.test
  # Handle the case where the test loader is missing.
  else:
    # Check if the validation loader exists.
    if (pipeline.val):
      # Print a warning about the missing test loader.
      fprint("Warning: Test loader not found. Using validation loader for testing.")
      # Assign the validation loader to the test loader variable.
      pipeline.test = pipeline.val
      # Add the assigned loader to the dictionary.
      availableLoaders["Test"] = pipeline.test
    # Handle the case where both test and validation loaders are missing.
    else:
      # Print a warning about missing test and validation loaders.
      fprint("Warning: Test loader not found. Using training loader for testing.")
      # Assign the training loader to the test loader variable.
      pipeline.test = pipeline.train
      # Add the assigned loader to the dictionary.
      availableLoaders["Test"] = pipeline.test
  # Ensure the validation loader is not None if both were initially missing.
  if (pipeline.val is None):
    # Assign the test loader to the validation loader variable.
    pipeline.val = pipeline.test
    # Add the assigned loader to the dictionary.
    availableLoaders["Val"] = pipeline.val
  # Plot class histograms for all splits.
  PlotClassHistograms(
    dataLoaders=availableLoaders,
    classNames=pipeline.classNames,
    outputDir=outputDir,
  )
  # Execute the training process.
  trainedModel, history, secondaryModel, configs = CreateFitViTModel(
    trainLoader=pipeline.train,
    valLoader=pipeline.val,
    modelName=modelName,
    numClasses=len(pipeline.classNames),
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
    useTopoWassersteinLoss=useTopoWassersteinLoss,
    epsilon=epsilon,
    lambdaTopo=lambdaTopo,
    lambdaContrastive=lambdaContrastive,
    usePretrainedCustomModels=usePretrainedCustomModels,
    pretrainedModelName=pretrainedModelName,
  )
  print(f"Training complete. Best model saved to {configs['ModelCheckpointPath']}")
  # Generate and save training history visualization plots.
  PlotTrainingHistory(history, outputDir=outputDir)
  # Load the best model from the checkpoint for evaluation.
  trainedModel.load_state_dict(torch.load(configs["ModelCheckpointPath"]))
  # Move the model to the device.
  trainedModel = trainedModel.to(device)
  # Evaluate the model on available splits and collect metrics.
  for loader, prefix in zip(
    [pipeline.test, pipeline.val, pipeline.train],
    ["Test", "Val", "Train"],
  ):
    # Check if the loader is not None.
    if (loader is not None):
      # Perform evaluation.
      EvaluateModel(
        model=trainedModel,
        dataLoader=loader,
        outputDir=outputDir,
        savePlots=True,
        classNames=pipeline.classNames,
        prefix=prefix,
        device=device,
        numClasses=len(pipeline.classNames),
        secondaryModel=secondaryModel,
      )
  # Print the pipeline completion summary for user confirmation.
  fprint(f"Pipeline Complete | Results saved to {outputDir}")


# Define the main execution block.
if (__name__ == "__main__"):
  # Check if CUDA is available and set the device accordingly.
  if (torch.cuda.is_available()):
    # Set the device to CUDA.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Print the device being used.
    fprint(f"Using device: {device}")
    # Calculate and print the GPU memory.
    gpuMemory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0
    # Print the GPU memory value.
    fprint(f"GPU Memory: {gpuMemory:.2f} GB")
  else:
    # Set the device to CPU if CUDA is not available.
    device = torch.device("cpu")
    # Print the device being used.
    fprint(f"Using device: {device}")
  # Import the argparse module for command-line interface.
  import argparse

  # Initialize the argument parser for command-line interface.
  parser = argparse.ArgumentParser(description="ViT Image Classification")
  # Add the command-line argument for the configuration file.
  parser.add_argument("--config", type=str, default="config.pytorch.yaml",
                      help="Path to the YAML or JSON configuration file")
  # Parse the command-line arguments into a namespace object.
  args = parser.parse_args()
  # Load the configuration from the file.
  config = LoadConfig(args.config)
  # Extract base parameters for the experiment grid.
  baseOutputDir = str(Path(config.get("OutputDir", "Results")).parent)
  # Extract the batch size from the configuration.
  batchSize = config.get("BatchSize", 16)
  # Extract lists or single values for the experiment grid.
  modelNames = config.get("ModelName", "StandardViT")
  # Extract the list of seeds for multi-trial execution.
  seeds = config.get("Seeds", [42])
  # Extract the optimizers from the configuration.
  optimizers = config.get("Optimizer", "Adam")
  # Extract the loss functions from the configuration.
  lossFunctions = config.get("LossFunction", "CrossEntropy")
  # Extract the pretrained custom models flag from the configuration.
  usePretrainedCustomModels = config.get("UsePretrainedCustomModels", False)
  # Extract the pretrained model name from the configuration.
  pretrainedModelName = config.get("PretrainedModelName", "vit_base_patch16_224")
  # Ensure the model names variable is a list for iteration.
  if (isinstance(modelNames, str)):
    # Convert a single string to a list.
    modelNames = [modelNames]
  # Ensure the optimizers variable is a list for iteration.
  if (isinstance(optimizers, str)):
    # Convert a single string to a list.
    optimizers = [optimizers]
  # Ensure the loss functions variable is a list for iteration.
  if (isinstance(lossFunctions, str)):
    # Convert a single string to a list.
    lossFunctions = [lossFunctions]
  # Ensure the seeds variable is a list for iteration.
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
          SetGlobalSeed(seed)
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
          experimentName = f"Exp-{folderModelName}-{optimizerName}-{batchSize}-{lossFunction}"
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
            RunCompletePipeline(
              dataDir=config.get("DataDir", "./data"),
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
              useAugmentation=config.get("UseAugmentation", False),
              useStainJitter=config.get("UseStainJitter", False),
              useBackgroundRemoval=config.get("UseBackgroundRemoval", False),
              useColorDeconvolution=config.get("UseColorDeconvolution", False),
              useTopoWassersteinLoss=config.get("UseTopoWassersteinLoss", True),
              epsilon=config.get("Epsilon", 0.1),
              lambdaTopo=config.get("LambdaTopo", 0.1),
              lambdaContrastive=config.get("LambdaContrastive", 0.1),
              usePretrainedCustomModels=usePretrainedCustomModels,
              pretrainedModelName=pretrainedModelName,
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
  fprint("All experiments completed.")
