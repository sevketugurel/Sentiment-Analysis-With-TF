# Sentiment Analysis with TensorFlow

This repository contains code for performing sentiment analysis on movie reviews using TensorFlow. The IMDb movie reviews dataset is used to train a neural network model.

## Getting Started

### Downloading the Dataset

The IMDb movie reviews dataset is downloaded and prepared using TensorFlow's utility functions.

```python
url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1"
import tensorflow as tf
dataset = tf.keras.utils.get_file(
    "aclImdb_v1",
    url,
    untar=True,
    cache_dir=".",
    cache_subdir=" "
)
import os
dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")
os.listdir(dataset_dir)
# ['imdbEr.txt', 'test', 'imdb.vocab', 'README', 'train']
train_dir = os.path.join(dataset_dir, "train")
os.listdir(train_dir)
# ['urls_unsup.txt', 'neg', 'urls_pos.txt', 'unsup', 'urls_neg.txt', 'pos', 'unsupBow.feat', 'labeledBow.feat']


### Loading the Dataset

The dataset is loaded and split into training, validation, and test sets.

```python
# Code for loading and splitting the dataset
batch_size = 32
seed = 42
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.20,
    subset="training",
    seed=seed
)


### Data Preprocessing

Text data is preprocessed, including custom standardization and vectorization.

```python
# Code for data preprocessing
import re
import string

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', '')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

from tensorflow.keras.layers import TextVectorization

max_features = 10000
sequence_length = 250
vectorization_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length
)


### Model Creation

A simple neural network model is created using TensorFlow's Sequential API.

```python
# Code for creating the neural network model
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features + 1, embedding_dim),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
model.summary()


### Model Compilation and Training

The model is compiled and trained on the training set.

```python
# Code for compiling and training the model
epochs = 10
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=tf.keras.metrics.BinaryAccuracy(threshold=0.0)
)

history = model.fit(
    raw_train_ds,
    validation_data=val_ds,
    epochs=epochs
)
# ...
```

### Model Evaluation

The trained model is evaluated on the test set to assess its performance.

```python
# Code for evaluating the model
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)


## Results

The model achieves a certain level of accuracy on the test set, as indicated in the README.

## Performance Optimization

The dataset is configured for performance using prefetch and cache to speed up training.

```python
# Code for configuring dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


## Plotting Training Metrics

Training metrics such as accuracy and loss are plotted over epochs to visualize model performance.

```python
# Code for plotting training metrics
import matplotlib.pyplot as plt

history_dict = history.history
epochs = range(1, len(history_dict['binary_accuracy']) + 1)

plt.plot(epochs, history_dict['loss'], 'bo', label='Training Loss')
plt.plot(epochs, history_dict['val_loss'], 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, history_dict['binary_accuracy'], 'bo', label='Training Accuracy')
plt.plot(epochs, history_dict['val_binary_accuracy'], 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


## Exporting the Model

The model is exported for later use, including the vectorization layer.

```python
# Code for exporting the model
export_model = tf.keras.Sequential([
    vectorization_layer,
    model,
    tf.keras.layers.Activation("sigmoid")
])
export_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=["accuracy"]
)

## Predicting New Data

The exported model is used to predict sentiment on new examples.

```python
# Code for predicting new data
examples = [
    "the movie was perfect.",
    "the movie was okay but was too long.",
    "the movie was awful."
]
export_model.predict(examples)



