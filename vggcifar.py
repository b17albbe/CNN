import os
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def unpickle(file, encoding='bytes'):
    with open(file, 'rb') as f:
        di = pickle.load(f, encoding=encoding)
    return di


print(os.listdir("cifar-10-batches-py"))

batches_meta = unpickle(f"cifar-10-batches-py/batches.meta", encoding='utf-8')
label_names = batches_meta['label_names']

batch_labels = []
batch_images = []

for n in range(1, 6):
    batch_dict = unpickle(f"cifar-10-batches-py/data_batch_{n}")
    # Add labels to the list of batch labels
    batch_labels.append(batch_dict[b'labels'])

    # Load the images, and resize them to 10000x3x32x32
    data = batch_dict[b'data'].reshape((10000, 3, 32, 32))
    # Modify axis to be 10000x32x32x3, since this is the correct order for keras
    data = np.moveaxis(data, 1, -1)
    batch_images.append(data)

labels = np.concatenate(batch_labels, axis=0)
images = np.concatenate(batch_images, axis=0)

test_dict = unpickle(f"cifar-10-batches-py/test_batch")
test_labels = np.array(test_dict[b'labels'])
test_images = test_dict[b'data'].reshape((10000,3,32,32))
test_images = np.moveaxis(test_images, 1, -1)

fig = plt.figure(figsize=(14,10))

for n in range(1, 29):
    fig.add_subplot(4, 7, n)
    img = images[n]
    plt.imshow(img)
    plt.title(label_names[labels[n]])
    plt.axis('off')
#plt.show()

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical

# We normalize the input according to the methods used in the paper
X_train = preprocess_input(images)
y_test = to_categorical(test_labels)

# We one-hot-encode the labels for training
X_test = preprocess_input(test_images)
y_train = to_categorical(labels)

from tensorflow.keras.applications.vgg16 import VGG16

model = VGG16(
    weights=None,
    include_top=True,
    classes=10,
    input_shape=(32,32,3)
)

# Expand this cell for the model summary
#model.summary()

from tensorflow.keras import optimizers

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'model.h5',
    monitor='val_accuracy',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

# Train the model
history = model.fit(
    x=X_train,
    y=y_train,
    validation_split=0.1,
    batch_size=256,
    epochs=5,
    callbacks=[checkpoint],
    verbose=1
)

with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_accuracy']].plot()

model.load_weights('model.h5')
train_loss, train_score = model.evaluate(X_train, y_train)
test_loss, test_score = model.evaluate(X_test, y_test)
print("Train Loss:", train_loss)
print("Test Loss:", test_loss)
print("Train F1 Score:", train_score)
print("Test F1 Score:", test_score)