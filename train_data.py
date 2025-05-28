import os
import tensorflow as tf  # for building and training the model
from tensorflow.python.keras.callbacks import ModelCheckpoint  # for saving the best model
import pandas as pd  # for creating and handling dataframe
import matplotlib.pyplot as plt  # for plotting graphs
import numpy as np  # for numerical computations
import seaborn as sns 
from sklearn.metrics import confusion_matrix  # for evaluating model performance with confusion matrix

print(tf.version.VERSION)  # print out the TensorFlow version to confirm it's installed properly

data_dir = "dataset"  

# gather all image paths and assign labels based on folder names
folders = os.listdir(data_dir)
paths = []
labels = []

for folder in folders:
    path_to_folder = os.path.join(data_dir, folder)
    files = os.listdir(path_to_folder)

    for file in files:
        paths.append(os.path.join(data_dir, folder, file))
        labels.append(folder.split(' ')[0])  # take first word in folder name as the label

# make a dataframe for easier analysis and debugging
data_frame = pd.DataFrame(data={"paths": paths, "labels": labels})

# show how many images are in each class
class_counts = data_frame["labels"].value_counts()
print(class_counts)

# plot class distribution
class_counts.plot(kind="barh", color="lightgreen")
plt.title("class distribution")
plt.xlabel("amount of images")
plt.ylabel("classes")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# set image size for model input
img_height = 350
img_width = 350

# split data into training and validation sets
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred", 
    label_mode="int", 
    class_names=None,  
    color_mode="rgb", 
    batch_size=32,  
    image_size=(img_height, img_width),  # resize all images to 350x350
    shuffle=True,
    seed=17,  
    validation_split=0.1,  # use 10% of data for validation
    subset="both",  # get both training and validation data
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# get class names
class_names = train_ds.class_names
print("class names:", class_names)

# display some sample images with their labels
images_batch, labels_batch = next(iter(train_ds))
labels_np = np.array([class_names[label] for label in labels_batch])
images_np = images_batch.numpy()
images_uint8 = images_np.astype(np.uint8)

plt.figure(figsize=(13, 13))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(images_uint8[i])
    plt.title(f"{labels_np[i]}", fontsize=9, pad=5)
    plt.axis("off")
plt.tight_layout()
plt.show()

# get total number of classes
label_map = train_ds.class_names
class_cnt = len(label_map)

# load ResNet50V2 as base model with pretrained weights
layers = tf.keras.layers
resnet = tf.keras.applications.ResNet50V2(
    include_top=False,  # exclude final classification layer
    weights='imagenet',  # use pretrained weights
    input_tensor=None,
    input_shape=(img_height, img_width, 3),
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
resnet.trainable = True  

# freeze all layers except the last few
for layer in resnet.layers[:-7]:
    layer.trainable = False

# define the complete model
model = tf.keras.models.Sequential([
    resnet,
    layers.GlobalAveragePooling2D(), 
    layers.Dense(256, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(class_cnt, activation='softmax')  
])

# set up a checkpoint to save the best performing model
savepoint = "model.h5"
checkpoint = ModelCheckpoint(savepoint,
                             monitor='accuracy', 
                             save_best_only=True,
                             verbose=1)

# compile the model with optimizer, loss and metric
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
history = model.fit(train_ds,
                    batch_size=32,
                    validation_data=val_ds,
                    epochs=10,
                    callbacks=[checkpoint])

# plot accuracy curves
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy over epochs')
plt.show()

# evaluate model on validation data
loss, accuracy = model.evaluate(val_ds)

# get predictions from model
predictions = model.predict(val_ds)

# convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.concatenate([y for x, y in val_ds], axis=0)
classes = val_ds.class_names

# compute confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes,
            yticklabels=classes)

plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.title("Confusion Matrix", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
