So here we will be perfroming sem 5 deep leanring experiments as a part of my B.Sc (Ds & AI).
total Experiments are 12
1. **Experiment 1:**
- **Title:**  **Handwritten Digit Classification using a Shallow Neural Network**
- **Objective:**  To classify handwritten digits (0-9) from the MNIST dataset using a shallow feedforward neural network.
- **Dataset:**  MNIST (Modified National Institute of Standards and Technology) dataset

Training set: 60,000 images, each 28x28 pixels (grayscale).

Test set: 10,000 images, each 28x28 pixels (grayscale).

Images represent digits from 0 to 9.
- **Model:**  Shallow Feedforward Neural Network (FNN)

Architecture:

Input Layer: Flattened 28x28 image (784 features).

Hidden Layer: One Dense layer with 128 neurons and ReLU activation.

Output Layer: One Dense layer with 10 neurons (for 10 classes) and Softmax activation.

Optimizer: Adam

Loss Function: sparse_categorical_crossentropy (suitable for integer labels)
- **Evaluation Metrics:**  Accuracy: The proportion of correctly classified images.

Loss: The value of the loss function during training and evaluation (sparse categorical crossentropy).
- **Hyperparameter Tuning:** Epochs: 10

Batch Size: 32

Learning Rate: Default for Adam optimizer (typically 0.001)
- **Experiment Design:**  The MNIST dataset is loaded and split into training and testing sets.

A shallow neural network model is constructed with an input layer to flatten the 2D image data, one hidden dense layer, and a dense output layer.

The model is compiled with the Adam optimizer, sparse_categorical_crossentropy loss, and accuracy as a metric.

The model is trained on the x_train and y_train data for 10 epochs with a batch size of 32.

The x_test and y_test data are used as validation data during training to monitor performance on unseen data.

After training, the model's performance (loss and accuracy) is evaluated on the independent x_test and y_test dataset.

Predictions are made on the x_test data, and the predicted class labels are extracted.

A visual inspection of the first 10 test images and their (implicitly) associated predictions is performed by displaying the images.
- **Expected Outcome:**  The model is expected to achieve a high accuracy on the test set, typically well over 90% for the MNIST dataset with this architecture, demonstrating its ability to accurately classify handwritten digits.
