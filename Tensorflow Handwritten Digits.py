# Jeremy Pretty
# CSC 510 - Crit 3 Tensorflow Handwritten Digits Test
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the ANN model
model = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Save model checkpoints for inference
model.save('mnist_ann_model')

# Load the saved model for inference
loaded_model = tf.keras.models.load_model('mnist_ann_model')

# Perform inference on test data
predictions = loaded_model.predict(x_test)

# Display the classification results
def display_result(index):
    plt.imshow(x_test[index], cmap='gray')
    plt.title(f"Predicted: {predictions[index].argmax()}, Actual: {y_test[index]}")
    plt.axis('off')
    plt.show()

# Show results for the first 5 images
for i in range(5):
    display_result(i)