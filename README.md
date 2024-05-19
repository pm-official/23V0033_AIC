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

# Question 03: Math Behind AI-ML Assignment


## 1. Comparing Distances on a 2D Grid

### Problem Statement
Assume a 2D Grid of constrained straight interlocking paths with well-defined start and end points. What metric should you use to compare distances between any two different paths? Why is it better than other metrics available? List out some of the other metrics.

### Solution

**Chosen Metric: Manhattan Distance**

**Explanation:**
To compare distances between any two different paths on a 2D grid with constrained straight interlocking paths, the most suitable metric is the **Manhattan Distance** (also known as Taxicab or L1 Distance).

**Advantages of Manhattan Distance:**
1. **Grid Alignment:** It perfectly aligns with the horizontal and vertical constraints of the grid, reflecting the actual travel distance.
2. **Computational Simplicity:** It is straightforward and less computationally intensive compared to other metrics.
3. **Relevance:** It accurately measures distance for grid-based paths, making it a practical choice for this scenario.

**Other Metrics:**
- **Euclidean Distance:** Measures the straight-line distance between two points. Not suitable for grid paths as it ignores the constraints of horizontal and vertical movements.
- **Chebyshev Distance:** Measures the maximum of the absolute differences of their coordinates. Useful for moves that can be in any direction but less intuitive for strictly horizontal/vertical paths.
## 2. Word Embeddings and Likeliness in High-Dimensional Space

### Problem Statement
While researching on Transformers, you learn about embeddings and how words are transformed into numerical vectors of high-dimensional space. What is the common criterion used to define the 'likeliness' mathematically between any two words?

### Solution

**Common Criterion: Cosine Similarity**

**Explanation:**
In high-dimensional spaces, Euclidean distances fall short in capturing the semantic similarity between word vectors. Instead, **Cosine Similarity** is used to measure the 'likeliness' between two words.

**Advantages of Cosine Similarity:**
1. **Direction Focused:** Measures the cosine of the angle between vectors, focusing on direction rather than magnitude.
2. **Normalized Measure:** Provides a normalized value between -1 and 1, making it easier to interpret similarity.

### Implementation
1. **Generate a corpus of words.**
2. **Convert these words into embeddings using a pre-trained model.**
3. **Calculate cosine similarity between word vectors.**
4. **Plot the embeddings in a 3D space using Plotly.js.**

#### Sample Python Code

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Sample word embeddings
words = ['happiness', 'sadness', 'success', 'failure']
embeddings = {
    'happiness': np.array([0.1, 0.8, 0.1]),
    'sadness': np.array([0.7, 0.1, 0.2]),
    'success': np.array([0.2, 0.9, 0.2]),
    'failure': np.array([0.8, 0.2, 0.3])
}

# Calculate Cosine Similarity
similarity_matrix = cosine_similarity(list(embeddings.values()))

# Plot embeddings in 3D
fig = px.scatter_3d(x=[emb[0] for emb in embeddings.values()],
                    y=[emb[1] for emb in embeddings.values()],
                    z=[emb[2] for emb in embeddings.values()],
                    text=words)
fig.show()
```

---
---

# Non-Technical Questions

### Q1. Timeline for Preparing for an Upcoming AI-Based Hackathon

### Problem Statement: Predicting House Prices

To prepare for an AI-based hackathon where we aim to predict house prices, I would follow a structured timeline, ensuring a comprehensive approach to both technical and organizational aspects. This preparation will span one month and will be divided into four key phases: Understanding the Problem, Data Preparation, Model Development, and Finalization.

### Week 1: Understanding the Problem and Gathering Resources
**Day 1-2: Problem Understanding**
- Review the problem statement and objectives.
- Identify the evaluation metrics (e.g., RMSE, MAE).
- Outline the project goals and deliverables.

**Day 3-4: Team Formation and Role Assignment**
- Assemble a team with diverse skills (data scientists, developers, domain experts).
- Assign roles and responsibilities to team members.
- Set up communication channels (Slack, Trello).

**Day 5-7: Dataset Acquisition and Exploration**
- Gather relevant datasets (e.g., Kaggle’s house prices dataset).
- Perform initial data exploration to understand the structure, features, and potential issues.
- Identify additional data sources that could enhance model performance (e.g., economic indicators, geographical data).

### Week 2: Data Preparation and Preprocessing
**Day 8-10: Data Cleaning**
- Handle missing values using imputation techniques.
- Remove duplicates and irrelevant features.
- Address outliers and anomalies.

**Day 11-13: Feature Engineering**
- Create new features that could improve model accuracy (e.g., age of the house, distance to city center).
- Perform feature selection to retain only the most impactful features.

**Day 14: Data Transformation**
- Normalize or standardize features as necessary.
- Encode categorical variables using techniques like one-hot encoding or label encoding.

### Week 3: Model Development and Evaluation
**Day 15-17: Baseline Model Development**
- Develop a simple baseline model (e.g., Linear Regression).
- Evaluate its performance using cross-validation.
- Document the baseline results for comparison.

**Day 18-21: Advanced Model Development**
- Experiment with various models (e.g., Decision Trees, Random Forests, Gradient Boosting, Neural Networks).
- Fine-tune hyperparameters using grid search or randomized search.
- Implement ensemble methods to combine the strengths of different models.

**Day 22-23: Model Evaluation**
- Evaluate models using the validation set.
- Compare model performance using evaluation metrics.
- Select the best-performing model based on the results.

### Week 4: Finalization and Presentation
**Day 24-26: Model Optimization**
- Perform model optimization techniques (e.g., feature scaling, hyperparameter tuning).
- Ensure the model generalizes well to unseen data.

**Day 27-28: Documentation and Visualization**
- Document the entire process, including data preprocessing, model development, and results.
- Create visualizations to illustrate key findings and model performance.

**Day 29-30: Presentation Preparation**
- Prepare a final presentation, including an overview of the problem, approach, and results.
- Practice the presentation with the team, ensuring clarity and coherence.

**Final Day: Hackathon Participation**
- Ensure all team members are aligned and ready.
- Present the model, its results, and potential future improvements.
- Be prepared to answer questions and demonstrate the robustness of the solution.

By following this structured approach, we ensure thorough preparation for the hackathon, maximizing our chances of success.

## Q2. Statement of Purpose (SOP) for Joining the Artificial Intelligence Community

### Introduction
Joining the AI Community at our college is an exciting opportunity for me to contribute to and learn from a group of like-minded individuals who share a common interest in AI technologies. I want to do My BTP and MTP in AI and therefore want to learn more about the technologies. ave done a BTP in computer vision which was related to activity recognition.

### Goals and Motivation
My primary goal in joining the AI Community is to deepen my understanding of AI and its applications. I am particularly interested in computer vision. By collaborating with peers, I hope to work on innovative projects that address real-world problems.

### Knowledge and Experience
I am proficient in using machine learning frameworks like TensorFlow and scikit-learn. Have done a BTP in computer vision which was related to activity recognition.

### Contributions to the Community
I aim to bring my enthusiasm and dedication to the AI Community. I am eager to participate in hackathons, workshops, and seminars, and contribute to ongoing projects.
