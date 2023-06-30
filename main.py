import matplotlib.pyplot as plt
import seaborn as sns
from imutils import paths

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import tensorflow as tf

from keras import datasets, layers, models
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, f1_score, recall_score, cohen_kappa_score, average_precision_score

import cv2

import numpy as np

directory = r'C:\Users\Sunkanmi-PC\Documents\School\University_files\500_level\Final-Year-Project\code\Training'

image_paths = list(paths.list_images(directory))

img_height = img_width = 120

# Training
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=32,
    image_size=(img_width, img_width),
    shuffle=True,
    validation_split=0.20,
    subset="training",
    seed=123,
    follow_links=False,
    crop_to_aspect_ratio=False
)

val_directory = r'C:\Users\Sunkanmi-PC\Documents\School\University_files\500_level\Final-Year-Project\code\Training'

# validation
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(img_height, img_width),
    shuffle=True,
    validation_split=0.20,
    subset="validation",
    seed=123,
    follow_links=False,
    crop_to_aspect_ratio=False
)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

# Cache training
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
 
# For training
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = layers.Rescaling(1./255)

num_classes = len(class_names)

# Reduce overfitting
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
  ]
)

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)

# filename='log.csv'
# history_logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[es]
)


testDir = r'C:\Users\Sunkanmi-PC\Documents\School\University_files\500_level\Final-Year-Project\code\Testing'

# test data
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='int',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=32
)

loss, accuracy = model.evaluate(test_data)

true_labels = []
for images, labels in test_data:
    true_labels.extend(labels.numpy())

true_labels = np.array(true_labels)

# Calculate predictions
predictions = model.predict(test_data)
predicted_labels = tf.argmax(predictions, axis=1)

# Calculate precision
precision = tf.keras.metrics.Precision()(true_labels, predicted_labels)

# Calculate recall
recall = recall_score(true_labels, predicted_labels, average='weighted')

# Calculate F1-score
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(true_labels, predicted_labels)

# Calculate Mean Average Precision (mAP)
# Assuming you have true labels and predicted probabilities for each class
# For mAP calculation, convert predicted probabilities to binary class predictions
binary_predictions = tf.one_hot(predicted_labels, depth=num_classes)
mAP = average_precision_score(tf.one_hot(true_labels, depth=num_classes), binary_predictions, average='macro')

# Calculate additional performance metrics

# 1. Balanced Accuracy
balanced_accuracy = recall_score(true_labels, predicted_labels, average='macro')

# Calculate confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)

# Display the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision.numpy())
print("Recall:", recall)
print("F1-score:", f1)
print("Cohen's Kappa:", kappa)
print("Mean Average Precision (mAP):", mAP)
print("Balanced Accuracy:", balanced_accuracy)

# Plot the confusion matrix
class_names = test_data.class_names

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


