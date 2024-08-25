import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

import os
from distutils.dir_util import copy_tree, remove_tree

from PIL import Image
from random import randint

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score

import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Flatten, LSTM, TimeDistributed, Reshape
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

print("TensorFlow Version:", tf.__version__)

base_dir = "D:\\140724\\EEG_EarlyPredection\\input\\"
root_dir = "D:\\140724\\EEG_EarlyPredection\\"
test_dir = base_dir + "test\\"
train_dir = base_dir + "train\\"
work_dir = root_dir + "dataset\\"

if os.path.exists(work_dir):
    remove_tree(work_dir)

os.mkdir(work_dir)
copy_tree(train_dir, work_dir)
copy_tree(test_dir, work_dir)
print("Working Directory Contents:", os.listdir(work_dir))

WORK_DIR = './dataset/'

CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

IMG_SIZE = 128
DIM = (IMG_SIZE, IMG_SIZE)

# Image Augmentation
ZOOM = [.99, 1.01]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"

work_dr = IDG(rescale=1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, 
              data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)

data_gen = work_dr.flow_from_directory(directory=WORK_DIR, target_size=DIM, batch_size=6400, shuffle=False)

def show_images(generator, y_pred=None):
    labels = dict(zip([0, 1, 2, 3], CLASSES))
    x, y = generator.next()
    
    plt.figure(figsize=(10, 10))
    if y_pred is None:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            idx = randint(0, len(x) - 1)
            plt.imshow(x[idx])
            plt.axis("off")
            plt.title("Class:{}".format(labels[np.argmax(y[idx])]))
    else:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(x[i])
            plt.axis("off")
            plt.title("Actual:{} \nPredicted:{}".format(labels[np.argmax(y[i])], labels[y_pred[i]]))

show_images(data_gen)

data, labels = data_gen.next()

sm = SMOTE(random_state=42)
data = data.reshape(-1, IMG_SIZE * IMG_SIZE * 3)
labels = np.argmax(labels, axis=1)

data, labels = sm.fit_resample(data, labels)

data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
labels = tf.keras.utils.to_categorical(labels, num_classes=4)

print(data.shape, labels.shape)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

inception_model = InceptionV3(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")

for layer in inception_model.layers[-30:]:
    layer.trainable = True

custom_inception_model = Sequential([
    inception_model,
    TimeDistributed(Flatten()),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    BatchNormalization(),
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    BatchNormalization(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    BatchNormalization(),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    BatchNormalization(),
    Dense(4, activation='softmax')
], name="inception_lstm_model")

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
rop_callback = ReduceLROnPlateau(monitor="val_loss", patience=3)

METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name='acc'),
    tf.keras.metrics.AUC(name='auc'),
    tfa.metrics.F1Score(num_classes=4)
]

custom_inception_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                               loss=tf.losses.CategoricalCrossentropy(),
                               metrics=METRICS)

custom_inception_model.summary()

EPOCHS = 50

history = custom_inception_model.fit(train_data, train_labels, 
                                     validation_data=(val_data, val_labels),
                                     callbacks=[early_stopping, rop_callback], 
                                     epochs=EPOCHS)

fig, ax = plt.subplots(1, 3, figsize=(30, 5))
ax = ax.ravel()

for i, metric in enumerate(["acc", "auc", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

# Save the Training and Validation Accuracy, AUC, and Loss Graph
plt.savefig('training_validation_metrics.png')

train_scores = custom_inception_model.evaluate(train_data, train_labels)
test_scores = custom_inception_model.evaluate(test_data, test_labels)

print("Training Accuracy: %.2f%%" % (train_scores[1] * 100))
print("Testing Accuracy: %.2f%%" % (test_scores[1] * 100))

pred_labels = custom_inception_model.predict(test_data)

def roundoff(arr):
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr

for labels in pred_labels:
    labels = roundoff(labels)

print(classification_report(test_labels, pred_labels, target_names=CLASSES))

pred_ls = np.argmax(pred_labels, axis=1)
test_ls = np.argmax(test_labels, axis=1)

conf_arr = confusion_matrix(test_ls, pred_ls)

# Save Confusion Matrix
plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(conf_arr, cmap='Greens', annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Alzheimer's Disease Diagnosis")
plt.xlabel('Prediction')
plt.ylabel('Truth')
plt.savefig('confusion_matrix_alzheimers.png')  # Save the Confusion Matrix as an image
plt.show()

# Balanced Accuracy Score and MCC
balanced_accuracy = round(BAS(test_ls, pred_ls) * 100, 2)
mcc_score = round(MCC(test_ls, pred_ls) * 100, 2)

print("Balanced Accuracy Score: {} %".format(balanced_accuracy))
print("Matthew's Correlation Coefficient: {} %".format(mcc_score))

# Generate the Precision-Recall Curve
precision = dict()
recall = dict()
average_precision = dict()

for i in range(len(CLASSES)):
    precision[i], recall[i], _ = precision_recall_curve(test_labels[:, i], pred_labels[:, i])
    average_precision[i] = average_precision_score(test_labels[:, i], pred_labels[:, i])

# Plot Precision-Recall curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(CLASSES)):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {CLASSES[i]} (AP = {average_precision[i]:0.2f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Alzheimer's Disease Diagnosis")
plt.legend(loc="best")
plt.savefig('precision_recall_curve_alzheimers.png')  # Save the Precision-Recall curve as an image
plt.show()

# Generate Detailed Metrics for Training and Testing Phases
metrics_data = {
    "Metric": ["Accuracy", "Balanced Accuracy", "MCC"],
    "Training": [f"{train_scores[1] * 100:.2f}%", "N/A", "N/A"],  # Replace "N/A" if you calculate these metrics for training data
    "Testing": [
        f"{test_scores[1] * 100:.2f}%",
        f"{balanced_accuracy}%",
        f"{mcc_score}%"
    ]
}

metrics_df = pd.DataFrame(metrics_data)
print(metrics_df)

# Save the metrics table as an image
import dataframe_image as dfi
dfi.export(metrics_df, "detailed_metrics_training_testing.png")

custom_inception_model_dir = work_dir + "alzheimer_inception_lstm_model_new"
custom_inception_model.save(custom_inception_model_dir, save_format='h5')
print(os.listdir(work_dir))

pretrained_model = tf.keras.models.load_model(custom_inception_model_dir)

plot_model(pretrained_model, to_file=work_dir + "model_plot.png", show_shapes=True, show_layer_names=True)
