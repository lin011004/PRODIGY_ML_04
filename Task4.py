#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define constants
GESTURE_CLASSES = ['10_down', '09_c', '08_palm_moved','07_ok','06_index','05_thumb','04_fist_moved','03_fist','02_l','01_palm']  # Add all gesture classes
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Adjust as needed

# Function to load images and labels
def load_data(root_dir):
    images = []
    labels = []
    for subject_folder in os.listdir(root_dir):
        subject_dir = os.path.join(root_dir, subject_folder)
        if os.path.isdir(subject_dir):
            for gesture_class in os.listdir(subject_dir):
                gesture_dir = os.path.join(subject_dir, gesture_class)
                if os.path.isdir(gesture_dir) and gesture_class in GESTURE_CLASSES:
                    for filename in os.listdir(gesture_dir):
                        img = cv2.imread(os.path.join(gesture_dir, filename), cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                        images.append(img)
                        labels.append(GESTURE_CLASSES.index(gesture_class))
    return np.array(images), np.array(labels)

# Load data
root_dir = "C:\\Users\\Lingesh\\Downloads\\archive (1)\\leapGestRecog"
images, labels = load_data(root_dir)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(GESTURE_CLASSES), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1), y_train, epochs=10, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1), y_test)
print("Test accuracy:", test_acc)

# Generate predictions
predictions = model.predict(X_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1))
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GESTURE_CLASSES, yticklabels=GESTURE_CLASSES)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, predicted_labels, target_names=GESTURE_CLASSES))

# Sample predictions
print("Sample Predictions:")
for i in range(10):
    print("True Label:", GESTURE_CLASSES[y_test[i]], "Predicted Label:", GESTURE_CLASSES[predicted_labels[i]])


# In[ ]:




