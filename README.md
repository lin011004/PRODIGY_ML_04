# Gesture Recognition with Convolutional Neural Network

This project involves building a Convolutional Neural Network (CNN) to recognize hand gestures from grayscale images. The dataset used is the Leap GestRecog dataset, which contains images of various hand gestures.

## Project Structure

The project consists of the following files and directories:

- `gesture_recognition.py`: Python script for training and evaluating the CNN model.
- `README.md`: This file.

## Requirements

- Python 3.6+
- numpy
- scikit-learn
- tensorflow
- opencv-python
- seaborn
- matplotlib

You can install the required libraries using pip:

```sh
pip install numpy scikit-learn tensorflow opencv-python seaborn matplotlib
```

## Code Overview

The script `Task4.py` performs the following steps:

1. **Define Constants**: Set the gesture classes and image dimensions.

    ```python
    GESTURE_CLASSES = ['10_down', '09_c', '08_palm_moved', '07_ok', '06_index', '05_thumb', '04_fist_moved', '03_fist', '02_l', '01_palm']
    IMG_HEIGHT, IMG_WIDTH = 64, 64
    ```

2. **Load Data**: Load images and labels from the dataset directory.

    ```python
    def load_data(root_dir):
        # Load images and labels from the dataset
    ```

3. **Preprocess Data**: Resize images, normalize pixel values, and split data into training and testing sets.

    ```python
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    ```

4. **Define the Model Architecture**: Build the CNN model using TensorFlow Keras.

    ```python
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(GESTURE_CLASSES), activation='softmax')
    ])
    ```

5. **Compile and Train the Model**: Compile the model with the Adam optimizer and sparse categorical cross-entropy loss. Train the model with a validation split.

    ```python
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1), y_train, epochs=10, validation_split=0.1)
    ```

6. **Evaluate the Model**: Evaluate the model on the test set and print the test accuracy.

    ```python
    test_loss, test_acc = model.evaluate(X_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1), y_test)
    print("Test accuracy:", test_acc)
    ```

7. **Generate Predictions**: Predict labels for the test set and compute the confusion matrix and classification report.

    ```python
    predictions = model.predict(X_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1))
    predicted_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(y_test, predicted_labels)
    ```

8. **Visualize Results**: Plot the confusion matrix and print the classification report.

    ```python
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=GESTURE_CLASSES, yticklabels=GESTURE_CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print("Classification Report:")
    print(classification_report(y_test, predicted_labels, target_names=GESTURE_CLASSES))
    ```

## Running the Code

1. Ensure that the dataset directory `leapGestRecog` is correctly set up as specified in the script.
2. Run the script `Task4.py`.

    ```sh
    python Task4.py
    ```

3. The script will load the dataset, train the CNN model, and display the confusion matrix and classification report.

## Acknowledgements

This project uses the Leap GestRecog dataset available on [Kaggle](https://www.kaggle.com/gti-upm/leapgestrecog).


