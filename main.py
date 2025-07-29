from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Input 
# Load MNIST dataset
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape # (60000, 28, 28)

import numpy as np
#print(np.array(x_train[0])/255.0)

def build_shallow_nn(input_shape, num_classes):
    model = Sequential([
          Input(shape=input_shape),
          Flatten(),
          Dense(128, activation='relu'),
          Dense(num_classes, activation='softmax')
          ])
    return model

input_shape = x_train.shape[1:] # (28, 28)

num_classes = len(set(y_train)) #10 classes from 0 to 9)

model = build_shallow_nn(input_shape, num_classes)


model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
# Evaluate Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

prediction=model.predict(x_test)

predict_class = np.argmax(prediction, axis=1)
print("predicted classes:", predict_class[:10])

import matplotlib.pyplot as plt
for i in range(10):
    plt.imshow(x_test[i], cmap = 'grey')
    plt.show()