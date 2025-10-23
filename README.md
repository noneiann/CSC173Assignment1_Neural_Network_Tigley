# CSC173 Activity 01 - Neural Network from Scratch

**Date:** October 09, 2025  
**Team:** Team Jinjin (Rey Iann V. Tigley)

## Project Overview

This project implements a simple neural network for binary classification using breast cancer diagnostic data. The network is built completely from scratch using only Python and NumPy, with no machine learning libraries. The goal is to deepen understanding of neural network fundamentals including forward propagation, loss computation, backpropagation, gradient descent training, and model evaluation.

## Data Preparation

I used the Breast Cancer Wisconsin Diagnostic dataset obtained from these sources:

- [Scikit-learn breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [UCI Machine Learning Repository (Breast Cancer Wisconsin Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

I selected two features from the dataset for the input layer of the network.

## Network Architecture

- Input layer: 2 neurons (corresponding to selected features)
- Hidden layer: 2 to 4 neurons, activation function: Sigmoid, ReLU, or Tanh
- Output layer: 1 neuron to produce binary classification output

## Implementation Details

- Weight and bias parameters initialized randomly.
- Forward propagation implements layer-wise computations with chosen activation functions.
- Loss computed using Mean Squared Error (MSE).
- Backpropagation calculates gradients of weights and biases.
- Parameters updated using gradient descent.
- Training performed for 500 to 1000 iterations.

## Results & Visualization

## Team Collaboration

I am the only member so I contributed to 100% of the output:

- Weight and bias initialization
- Forward propagation coding
- Loss function implementation
- Backpropagation and gradient computation
- Training loop and visualization

## How to Run

1. Clone the GitHub repository:
   ```
   git clone [repository_url]
   ```
2. Open the Jupyter notebook or Colab file.
3. Run all cells sequentially.
4. Explore training loss plot and decision boundary visualizations.

## Summary

This activity provided hands-on experience in constructing a neural network from scratch without the use of high-level machine learning frameworks. I independently developed the model, analyzed its training behavior through visualizations, and demonstrated a clear understanding of fundamental AI concepts through both coding and documentation.

The activity helped me both understand how machine learning models work on a deeper, more theoretical level, while also helping me hone my algorithmic programming skills, where I had to code lots of mathematical formulas into python.

Video: [link](https://drive.google.com/file/d/1nFaIlOZlFfzdhq2ljocpdHaRPBysoyij/view?usp=sharing)
