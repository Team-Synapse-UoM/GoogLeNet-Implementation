# GoogleNet Implementation

This repository demonstrates the training of a GoogLeNet model on the MNIST dataset using PyTorch. The dataset is resized and normalized to suit the GoogLeNet architecture, and the implementation achieves high accuracy.

---

## GoogLeNet Architecture

GoogLeNet, also known as Inception v1, is a convolutional neural network architecture that uses inception modules to capture multi-scale features efficiently. This implementation uses the standard GoogLeNet variant.
![alt text](Googlent.png)

---

## Project Steps

### 1. Import Required Libraries

Set up the environment using the following commands:

```bash
conda create --name pytorch_env python=3.11.9 --file requirements.txt
conda activate pytorch_env
```

### 2. Load the MNIST Dataset

- The dataset is loaded using `torchvision.datasets.MNIST`.
- Images are resized to 32x32 (to fit GoogLeNet requirements).
- Images are normalized and converted to tensors.

### 3. Split the Dataset

The dataset is split as follows:

- **Training Set**: 42,000 samples
- **Validation Set**: 14,000 samples
- **Test Set**: 14,000 samples

### 4. Load Pre-trained GoogLeNet

- GoogLeNet is used with modifications:
  - Adjusted for single-channel (grayscale) input.
  - Output layer configured for 10 classes.
- The model is moved to GPU if available.

### 5. Define Loss Function and Optimizer

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam with a learning rate of 0.001

### 6. Train the Model

The model is trained for 20 epochs using the following key steps:

- The training set is passed through the model to compute predictions.
- The loss is calculated using CrossEntropyLoss.
- Gradients are computed and weights are updated using the Adam optimizer.

---

## Results

### Test Accuracy

- The model achieves a test accuracy of **99.04%**.

### Training and Validation Losses

Plots showing the training and validation loss over epochs:

![Loss vs Epochs](output.png)

### Confusion Matrix

A heatmap of the confusion matrix shows the model's performance across all classes:

![Confusion Matrix](matrix.png)

---

## Requirements

- Python 3.11.9
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL (Pillow)

For a detailed list of dependencies, refer to `requirements.txt`.

---

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [GoogLeNet Paper](https://arxiv.org/abs/1409.4842)
