import tensorflow as tf
import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Download and load the wine dataset
url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

urllib.request.urlretrieve(url_red, "winequality-red.csv")
urllib.request.urlretrieve(url_white, "winequality-white.csv")

red_wine = pd.read_csv("winequality-red.csv", delimiter=";")
white_wine = pd.read_csv("winequality-white.csv", delimiter=";")

# Add a 'type' column to the dataset
red_wine["type"] = "red"
white_wine["type"] = "white"

# Combine red and white wine datasets
wine_data = pd.concat([red_wine, white_wine])

# Encode the wine type labels
label_encoder = LabelEncoder()
wine_data["type"] = label_encoder.fit_transform(wine_data["type"])

# Split dataset into input features (X) and target labels (y)
X = wine_data.drop(["type", "quality"], axis=1)
y = wine_data["type"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the ANN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model and its checkpoints
model.save("wine_classifier.h5")

# Load the saved model
loaded_model = tf.keras.models.load_model("wine_classifier.h5")

# Perform inference on the test set
y_pred = (loaded_model.predict(X_test) > 0.5).astype("int32")

# Visualize the results
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
print(comparison_df.head(20))
