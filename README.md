# cs231n-assignment2-2025
# Stanford University CS231n – Assignment 2 Solutions (2025)

This repository contains my solutions for the
CS231n: Deep Learning for Computer Vision Assignment 2 (https://cs231n.github.io/assignments2025/assignment2/)

- Training convolutional networks on CIFAR-10
- Implementing Batch Normalization and Dropout
- Building ConvNets in PyTorch
- Implementing an RNN/LSTM-based image captioning model in PyTorch

All core parts are implemented by hand following the official starter code.


## Academic Integrity Notice

These notebooks are shared **for learning and reference only**.
If you are currently taking CS231n, **do not copy** this work or submit it as your own.  
Use it only to understand the structure of the assignment and compare ideas after you’ve done your own implementation.


## Contents

The repository includes the following Jupyter notebooks:

### `BatchNormalization.ipynb`
- Implements batch normalization forward and backward passes from scratch.
- Verifies gradients with numerical gradient checks.
- Runs experiments varying the weight initialization scale and visualizes:
   Best training / validation accuracy vs. weight scale
   Final training loss vs. weight scale
- Shows how batch normalization makes training more stable and less sensitive to initialization.

### `Dropout.ipynb`
- Implements inverted dropout forward and backward passes.
- Uses gradient checking to confirm correctness.
- Compares training and validation accuracy with and without dropout and its impact on overfitting.

### `ConvolutionalNetworks.ipynb`
- Implements key CNN building blocks in NumPy:
   Naive convolution forward and backward
   Max-pooling forward and backward
- Combines them into a three-layer ConvNet:
  `conv – relu – 2x2 max pool – affine – relu – affine – softmax`
- Trains the ConvNet on CIFAR-10 and evaluates loss and accuracy over time.

### `PyTorch.ipynb`
- Reimplements ConvNets in PyTorch in several stages:
  - Barebones ConvNet using only tensor operations and manual forward pass.
  - ConvNet using the `nn.Module` API.
  - ConvNet using the `nn.Sequential` API with Nesterov momentum.
- Trains these models on CIFAR-10, comparing:
  - Different architectures
  - Different optimizers (SGD, SGD+momentum, Adam, etc.)
- Includes an open-ended experiment to reach ≥ 70% validation accuracy within 10 epochs using a deeper ConvNet with batch normalization, dropout, and a modern optimizer.

### `RNN_Captioning_pytorch.ipynb`
- Implements core recurrent components in PyTorch:
  - Vanilla RNN step and sequence forward (`rnn_step_forward`, `rnn_forward`)
  - LSTM step and sequence forward (`lstm_step_forward`, `lstm_forward`)
  - Word embeddings and temporal affine layers
- Builds a CaptioningRNN class that:
  - Maps image features to an initial hidden state
  - Uses an RNN or LSTM to generate captions word-by-word
  - Computes sequence loss with a temporal softmax over the vocabulary
- Implements a sampling procedure to generate captions at test time from image features.


## Environment & Requirements

The code is based on the official CS231n 2025 assignments and uses the standard Python scientific stack plus PyTorch.


### Recommended environment

- Python 3.8+
- Google Colab

### Python packages

- `numpy`
- `matplotlib`
- `torch`
- `torchvision` (for CIFAR-10 / image utilities)
- `tqdm` (optional, for progress bars)

You can install the core dependencies via:

```bash
pip install numpy matplotlib torch torchvision
