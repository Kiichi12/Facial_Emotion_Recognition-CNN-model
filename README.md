# ProblemStatement 
-Building a CNN model for facial emotion recognition so as to classify a given image into any one ofthe 7 expressions:- Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
-To evaluate the model's performance using f1-score, confusion matrix etc.
# Approach:
-The FER 2013 dataset was used for this problem
## Data Preprocessing
  - Rescaled pixel values to the range `[0, 1]` using `ImageDataGenerator` with `rescale=1./255`.
  - Augmented the training data to improve model generalization. Augmentations included:
    - Randomly zoomed images by up to 30%.
    - Used 'nearest' for filling in augmented regions.

## Model Architecture
- Designed a convolutional neural network (CNN) with the following layers:
  - Convolutional Layers: Extract spatial features using filters of size `(3x3)` with ReLU activation.
  - Pooling Layers: Reduced spatial dimensions using max pooling `(2x2)`.
  - Dropout Layers: Added dropout to prevent overfitting.
  - Fully Connected Layers: Flattened features and passed them through dense layers for classification.
  - Output Layer: Used a dense layer with a softmax activation function for the 7-class prediction.

## Model Training
- Used categorical cross-entropy for multi-class classification.
- Used Adam optimizer for efficient gradient updates.
- Monitored loss and accuracy during training and validation.
- Hyperparameters:
  - Batch size: 32
  - Input size: `(40x40x1)` (grayscale images)
  - Epochs: 30

## Evaluation
- Evaluated the trained model on the test dataset using:
  - Confusion Matrix: To analyze predictions for each class.
  - Classification Report: Precision, recall, F1-score for each class.
- Visualized misclassified and correctly classified images for further insights.

# Challenges
- Challenge: The dataset was imbalanced (e.g., fewer examples for 'Disgust' class).
  - Solved it by applying data augmentation to artificially balance the classes and improve model learning.

# Results:
- Achieved competitive performance with 0.63(F1 accuracy score) on the test dataset.
- Observed improvement in misclassified cases after data augmentation and hyperparameter tuning.
