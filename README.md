# Handwritten Digit Classification using a Shallow Neural Network

*   **Objective:** To classify handwritten digits (0-9) from the MNIST dataset using a shallow feedforward neural network.
*   **Dataset:** MNIST (Modified National Institute of Standards and Technology) dataset
    *   Training set: 60,000 images, each 28x28 pixels (grayscale).
    *   Test set: 10,000 images, each 28x28 pixels (grayscale).
*   **Model:** Shallow Feedforward Neural Network (FNN)
    *   **Architecture:**
        *   Input Layer: Flattened 28x28 image (784 features).
        *   Hidden Layer: One Dense layer with 128 neurons and ReLU activation.
        *   Output Layer: One Dense layer with 10 neurons (for 10 classes) and Softmax activation.
    *   **Optimizer:** Adam
    *   **Loss Function:** sparse_categorical_crossentropy
*   **Evaluation Metrics:**
    *   Accuracy
    *   Loss
*   **Hyperparameter Tuning:**
    *   Epochs: 10
    *   Batch Size: 32
    *   Learning Rate: Default for Adam optimizer (typically 0.001)
*   **Expected Outcome:** The model is expected to achieve a high accuracy on the test set, typically well over 90%.