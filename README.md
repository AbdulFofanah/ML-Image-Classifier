# Oxford-IIIT Pet Image Classifier

A custom convolutional neural network built in PyTorch for 37-class image classification on the Oxford-IIIT Pet dataset.

The model combines RGB imagery with the dataset's trimap segmentation information as a fourth input channel, alongside synchronised data augmentation, dataset-specific normalisation and scheduled optimisation.

**Official test accuracy: 73.62%**  
**Coursework mark: 92/100**

## Overview

This project was developed for the University of York's **Intelligent Systems: Machine Learning & Optimisation** module.

The objective was to design, train and evaluate a neural network classifier on the Oxford-IIIT Pet dataset under a maximum training budget of 30 epochs.

The final model was trained for 29 epochs and classifies images across **37 cat and dog breeds**.

## Model Architecture

The final model is a custom PyTorch convolutional neural network containing **4,366,053 parameters**.

The architecture uses:

- Convolutional layers for feature extraction
- Batch normalisation
- ReLU activations
- Max pooling
- Adaptive average pooling
- Dropout
- A fully connected classification layer with 37 outputs

The network progressively increases feature depth from 32 to 64, 256 and 512 channels before global feature aggregation and final classification.

## Data Pipeline

The network uses a **four-channel input** consisting of:

- Three RGB image channels
- A binary pet mask derived from the official Oxford-IIIT Pet trimap segmentation

Training preprocessing includes:

- Random resized cropping
- Random horizontal flipping
- Synchronised transformations of the RGB image and trimap mask
- Tensor conversion
- Dataset-specific RGB normalisation

The dataset was split into:

- **3,496 training images**
- **184 validation images**
- The official Oxford-IIIT Pet test set for final evaluation

## Training

The final training configuration used:

- **Framework:** PyTorch
- **Optimiser:** AdamW
- **Loss:** Cross-entropy with label smoothing
- **Learning-rate scheduler:** OneCycleLR
- **Batch size:** 16
- **Training epochs:** 29
- **Classes:** 37

A variable learning-rate schedule was used across training, reaching a maximum learning rate of `2e-3` before progressively decreasing towards the end of training.

## Results

| **Metric** | **Result** |
| --- | ---: |
| Official test accuracy | **73.62%** |
| Coursework mark | **92/100** |
| Model parameters | **4,366,053** |
| Training images | **3,496** |
| Validation images | **184** |
| Classes | **37** |
| Training epochs | **29** |

## Key Technical Features

- Custom CNN architecture implemented in PyTorch
- 37-class image classification
- Four-channel RGB + trimap-mask input
- Synchronised image and segmentation-mask augmentation
- Dataset-specific normalisation
- AdamW optimisation
- Label-smoothed cross-entropy loss
- OneCycleLR learning-rate scheduling
- Reproducible training and evaluation pipeline

## Technologies

**Python · PyTorch · Torchvision · NumPy · scikit-learn**

## Academic Context

This project formed the coursework component of the University of York **Intelligent Systems: Machine Learning & Optimisation** module.

The module covered machine learning, neural networks, gradient-based optimisation, linear and logistic regression, regularisation, backpropagation and the mathematical foundations of machine learning.

The coursework received **92/100**.

## Academic Integrity and Portfolio Use

This repository contains my completed coursework implementation and is published as a portfolio archive for recruitment and professional development.

It is not intended to provide solutions for current or future assessed work. Students completing related coursework should follow their institution's academic-integrity requirements and produce their own independent work.
