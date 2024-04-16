# CIFAR-10 Image Classification with CNN

- **Author:** Arshia Goshtasbi

- **GitHub:** [@Arshiagosh](https://github.com/Arshiagosh)

This repository contains a Python code implementation for training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset for image classification tasks. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Dataset

The CIFAR-10 dataset classes are:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Code Overview

The code covers the following steps:

1. **Data Preprocessing**:

- Loading the CIFAR-10 dataset from the Keras library
- Splitting the original training data into training and validation sets
- Normalization of image pixel values
- One-hot encoding of image labels
- Data augmentation techniques (rotation, shifting, flipping, zooming, brightness adjustment, shear, and channel shifting)

2. **CNN Model Architecture**:

- Designing a Convolutional Neural Network (CNN) using the Keras Sequential model
- The architecture includes convolutional layers with 3x3 kernels, batch normalization, max pooling, and dropout layers for regularization

3. **Model Training**:

- Compiling the model with the Adam optimizer, categorical cross-entropy loss, and accuracy metric
- Setting up callbacks for learning rate reduction on plateau and early stopping
- Training the model using the prepared dataset and data augmentation generator for a specified number of epochs

4. **Learning Curve Visualization**:

- Plotting the training and validation loss, as well as the training and validation accuracy over epochs

5. **Model Evaluation**:

- Evaluating the trained model on the unseen test data
- Printing the test accuracy and loss
- Obtaining predicted labels for the test data
- Computing and visualizing a confusion matrix to analyze the model's performance across different classes

## Requirements

To run this code, you need to have the following libraries installed:

- Python (version 3.x)
- NumPy
- Matplotlib
- Scikit-learn
- Keras
- Seaborn

## Usage

1. Clone this repository to your local machine.
2. Install the required libraries mentioned above.
3. Run the `main.py` script to execute the code.

The code will load the CIFAR-10 dataset, preprocess the data, train the CNN model, visualize the learning curves, and evaluate the model's performance on the test data. The final result will include the test accuracy, test loss, and a confusion matrix visualized as a heatmap.

## Contributing

Contributions are welcome! If you find any issues or want to enhance the code, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
