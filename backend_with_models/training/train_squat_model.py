# Data manipulation
import pandas as pd

# Scikit-learn for preprocessing and model selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# TensorFlow and Keras for deep learning
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Dataset
dataset_path = "dataset/squat_dataset.csv"  # Path to dataset
data = pd.read_csv(dataset_path)

# 2. Preprocess Data
X = data.drop(columns=["Label"]).values  # Drop the "Label" column (features)
y = data["Label"].values  # Keep the "Label" column (target)

# Encode labels (e.g., "Good squat" -> 0, "Bad squat" -> 1, "No squat" -> 2)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build the Model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation="relu"),
    Dense(32, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")  # Output layer
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 4. Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 5. Save the Model
model.save("training/models/squat_model.h5")
print("Model trained and saved!")

# Optional: Evaluate Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")