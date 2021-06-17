#!/usr/bin/env python
# coding: utf-8

# ### **Use this Notebook on Google Colab Free GPU**
# Go to Google Drive, upload this notebook and dataset folders. Rigth click notebook, choose **Open with -> Google Colab**
# <br>
# To activate GPU in Colab, go to **Runtime -> Change Runtime Type**. Under **Hardware accelerator** choose **GPU**

# ### Connect to Google Drive to access Dataset

# In[1]:


from google.colab import drive
drive.mount('/content/drive')
get_ipython().run_line_magic('cd', "'/content/drive/My Drive/Tb-Detection'")


# ### Import all dependencies

# In[2]:


from __future__ import print_function, division
from builtins import range, input

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
import pandas as pd
from PIL import Image
from os import listdir
from os.path import splitext
import cv2 


# ### Define Few Parameters

# In[3]:


#define size to which images are to be resized
IMAGE_SIZE = [224, 224] # feel free to change depending on dataset

# training config:
epochs = 5
batch_size = 32

pngs = glob('data/chest/TB./*.png')

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)

pngs = glob('data/chest/Normal./*.png')

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)
TB_path = 'data/chest/TB'
Normal_path = 'data/chest/Normal'     

# Use glob to grab images from path .jpg or jpeg
TB_files = glob(TB_path + '/*')
Normal_files = glob(Normal_path + '/*')


# ### Fetch Images and Class Labels from Files (This might take a while)

# In[4]:


# Preparing Labels
TB_labels = []
Normal_labels = []

TB_images=[]
Normal_images=[]

import cv2 

for i in range(len(TB_files)):
  image = cv2.imread(TB_files[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(224,224))
  TB_images.append(image)
  TB_labels.append('TB')
for i in range(len(Normal_files)):
  image = cv2.imread(Normal_files[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(224,224))
  Normal_images.append(image)
  Normal_labels.append('Normal')


# ### Visualize First 40 Images from Data set

# In[5]:


# look at a random image for fun
def plot_images(images, title):
    nrows, ncols = 5, 8
    figsize = [10, 6]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=(1, 1, 1))

    for i, axi in enumerate(ax.flat):
        axi.imshow(images[i])
        axi.set_axis_off()

    plt.suptitle(title, fontsize=24)
    plt.tight_layout(pad=0.2, rect=[0, 0, 1, 0.9])
    plt.show()
plot_images(TB_images, 'TB patient Chest X-ray')
plot_images(Normal_images, 'Normal patient  Chest X-ray')


# ### **Normalization**
# #### Model takes images in the form of array of pixels. Hence convert into array and *normalize*

# In[6]:


# Convert to array and Normalize to interval of [0,1]
TB_images = np.array(TB_images) / 255
Normal_images = np.array(Normal_images) / 255


# ### **Train Test Split**

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

# split into training and testing
TB_x_train, TB_x_test, TB_y_train, TB_y_test = train_test_split(
    TB_images, TB_labels,test_size=0.2)
Normal_x_train, Normal_x_test, Normal_y_train, Normal_y_test = train_test_split(
    Normal_images, Normal_labels,test_size=0.2)

X_train = np.concatenate((Normal_x_train, TB_x_train), axis=0)
X_test = np.concatenate((Normal_x_test, TB_x_test), axis=0)
y_train = np.concatenate((Normal_y_train, TB_y_train), axis=0)
y_test = np.concatenate((Normal_y_test, TB_y_test), axis=0)

# make labels into categories - either 0 or 1
y_train = LabelBinarizer().fit_transform(y_train)
y_train = to_categorical(y_train)

y_test = LabelBinarizer().fit_transform(y_test)
y_test = to_categorical(y_test)


# ### Visualize a few images from Training and Test sets

# In[8]:


plot_images(TB_x_train, 'X_train')
plot_images(TB_x_test, 'X_test')
# y_train and y_test contain class lables 0 and 1 representing COVID and NonCOVID for X_train and X_test


# ### **Building and Visualizing model**

# In[9]:


vggModel = VGG19(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

outputs = vggModel.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(2, activation="softmax")(outputs)

model = Model(inputs=vggModel.input, outputs=outputs)

for layer in vggModel.layers:
    layer.trainable = False

model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)


# In[10]:


model.summary()


# ### **Image Augmentation**
# To train on images at different positions, angles, flips, etc

# In[11]:


train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


# ### **Training the model**

# In[12]:


history = model.fit(train_aug.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) / 32,
                    steps_per_epoch=len(X_train) / 32,
                    epochs=500)


# In[13]:


model.save('vgg_chest.h5')


# In[14]:


model.save_weights('vggweights_chest.hdf5')


# In[15]:


model = load_model('vgg_chest.h5')


# ### **Making Predicions**

# In[16]:


y_pred = model.predict(X_test, batch_size=batch_size)


# ### Visulaizing First 10 predictions

# In[17]:


prediction=y_pred[0:10]
for index, probability in enumerate(prediction):
  if probability[1] > 0.5:
        plt.title('%.2f' % (probability[1]*100) + '% TB')
  else:
        plt.title('%.2f' % ((1-probability[1])*100) + '% NORMAL')
  plt.imshow(X_test[index])
  plt.show()


# In[18]:


# Convert to Binary classes
y_pred_bin = np.argmax(y_pred, axis=1)
y_test_bin = np.argmax(y_test, axis=1)


# ### Plot ROC Curve

# In[19]:


fpr, tpr, thresholds = roc_curve(y_test_bin, y_pred_bin)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for our model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)


# ### Plot Confusion Matrix

# In[20]:


def plot_confusion_matrix(normalize):
  classes = ['TB','NORMAL']
  tick_marks = [0.5,1.5]
  cn = confusion_matrix(y_test_bin, y_pred_bin,normalize=normalize)
  sns.heatmap(cn,cmap='plasma',annot=True)
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

print('Confusion Matrix without Normalization')
plot_confusion_matrix(normalize=None)

print('Confusion Matrix with Normalized Values')
plot_confusion_matrix(normalize='true')


# ### **Classification Report**

# In[21]:


from sklearn.metrics import classification_report
print(classification_report(y_test_bin, y_pred_bin))


# ### **Accuracy and Loss Plots**

# In[22]:


plt.figure(figsize=(10,10))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('vgg_chest_accuracy.png')
plt.show()


# In[23]:


plt.figure(figsize=(10,10))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('vgg_chest_loss.png')
plt.show()


# In[23]:





# In[23]:




