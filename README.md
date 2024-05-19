# Question 01
# CNN Model for Object Detection

## Objective
Develop a CNN model for object detection using TensorFlow and Keras. Use the CIFAR-10 dataset for training and testing the model.

## Steps to Run the Code on Your Local Machine

### 1. Install Necessary Libraries
Ensure you have Python installed. Then install the required libraries using pip:

```bash
pip install tensorflow matplotlib
```

### 2. Load and Preprocess Data
The CIFAR-10 dataset is loaded directly from TensorFlow/Keras datasets. It includes 60,000 32x32 color images in 10 classes, with 6,000 images per class.

### 3. Construct the CNN Model
Build a Convolutional Neural Network using TensorFlow/Keras with the following architecture:
- Three convolutional layers with ReLU activation
- Two max pooling layers
- A flattening layer
- Two dense layers

### 4. Compile and Train the Model
Compile the model using Adam optimizer and sparse categorical crossentropy loss. Train the model for 10 epochs.

### 5. Evaluate the Model
Evaluate the model on the test set and display the accuracy.

### 6. Save the Model
Save the trained model for future use.

## Code Documentation and Thought Process

### Importing Libraries

```python
# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```

### Loading the CIFAR-10 Dataset

```python
# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizing the images to [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Displaying the first 5 images and their labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
```

### Constructing the CNN Model

```python
# Building the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

### Training the Model

```python
# Training the model
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Plotting the training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
```

### Evaluating the Model

```python
# Evaluating the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test Accuracy: {test_acc}')
```

### Saving the Model

```python
# Saving the trained model
model.save('cifar10_cnn_model.h5')
```

## Work Analysis

### Challenges Faced
1. **Data Preprocessing:** Adjusting the dataset format and normalizing images to fit the model input requirements.
2. **Model Complexity vs Training Time:** Balancing between a simple model for quick training and a complex model for better performance within the 10-minute constraint.

### Improvements and Future Work
1. **Advanced Architectures:** Using models like ResNet or Inception for better accuracy.
2. **Data Augmentation:** Implementing data augmentation to enhance model robustness.
3. **Hyperparameter Tuning:** Experimenting with different learning rates, batch sizes, and epochs for optimal performance.

### Final Comments
This project demonstrates the implementation of a CNN for object detection using the CIFAR-10 dataset. By balancing complexity and efficiency, we trained a model within a short time frame that achieved satisfactory accuracy. This foundation can be expanded upon with more advanced techniques and larger datasets for improved results.

### References
- TensorFlow Documentation: https://www.tensorflow.org/tutorials
- Keras Documentation: https://keras.io/api/
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

By documenting my thought process and handling errors through trial and error, debugging, and referencing official documentation, I gained practical experience in developing and evaluating a CNN model.

This README provides a comprehensive guide to understand and reproduce the project on any local machine.

---
---
