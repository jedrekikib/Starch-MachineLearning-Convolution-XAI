# Microscopic Starch Image Classification

This repository contains research code for classifying 5 types of starch (corn, wheat, rice, tapioca, potato) from microscopic images using a 5-Fold Cross-Validation approach.

## Repository Structure

* **`DataSetCreation.ipynb`**: Handles data preprocessing, dataset splitting (5 folds via `GroupShuffleSplit`), and image augmentation using the `albumentations` library.
* **`StarchML_Experiments.ipynb`**: Contains the full training and evaluation pipeline:
* **Classic ML**: Feature extraction using LBP (Local Binary Patterns) + PCA, optimized with Grid Search for models like SVM, RF, and MLP.
* **Deep Learning**: Training from scratch and transfer learning using architectures like Custom CNN, ResNet18, AlexNet, and DenseNet121.
* **Explainable AI (XAI)**: Includes Grad-CAM visualizations to interpret CNN decision-making.





## Main Requirements

* `torch`, `torchvision`
* `scikit-learn`
* `opencv-python`
* `albumentations`
* `pytorch-grad-cam`
