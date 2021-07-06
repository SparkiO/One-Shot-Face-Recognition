# %% [markdown]
# Set Variables
IMG_SHAPE = (160, 160)

import sys
if len(sys.argv) < 2:
    print('Usage: python FaceRecognition1.py <train_directory> <test_directory>')
    sys.exit(1)
train_path = sys.argv[1]
test_path = sys.argv[2]

# %% [markdown]
# Load Dependencies
print('* Importing dependencies *')

import os
import glob
import numpy as np
import cv2
from PIL import Image, ImageOps
from scipy.io import savemat
from tensorflow.keras.models import load_model
from sklearn.svm import SVC

# %% [markdown]
# Load Train Dataset
print('* Augumenting train samples *')

def augument_1(img):
    img_resized = img.resize(IMG_SHAPE, Image.ANTIALIAS)
    img_flipped = ImageOps.mirror(img_resized)
    augments = []
    augments.append(np.asarray(img_resized))
    augments.append(np.asarray(img_flipped))
    return augments
    
train_x = []
train_y = [os.path.basename(img_path) for img_path in glob.glob(train_path + '*')]
for label in train_y:
    filename = glob.glob(train_path + label + '/*')[0]
    with Image.open(filename).convert('RGB') as img:
        augments = augument_1(img)
        train_x.append(augments)

# %% [markdown]
# Load FaceNet Model
print('* Loading FaceNet model *')

model_facenet = load_model('facenetnew.h5', compile=False)

# %% [markdown]
# Load Face Detector

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def get_face(img):
    face = detector.detectMultiScale(img, 1.1, 4)
    try:
        x, y, w, h = face[0]
    except:
        return img
    x = (abs(x), abs(x)+w)
    y = (abs(y), abs(y)+h)
    
    face = img[y[0]:y[1], x[0]:x[1]]
    img_face = Image.fromarray(face, 'RGB').resize(IMG_SHAPE)
    return np.asarray(img_face)

def get_embeddings(face):
  face = face.astype('float32')
  face = (face - face.mean()) / face.std()
  prediction = model_facenet.predict(face)[0]
  return prediction

# %% [markdown]
# Detect Faces
print('* Encoding train data *')

train_x = np.asarray(train_x)
train_x = np.asarray([get_embeddings(x) for x in train_x])


# %% [markdown]
# Train SVC Model
print('* Training SVC model *')

model_svc = SVC(kernel='linear', probability=True)
model_svc.fit(train_x, train_y)

# %%
del train_x
del train_y

# %% [markdown]
# Predict Samples
print('* Predicting test samples *')

def predict(img):
    face = get_face(np.expand_dims(img, axis=0))
    embeding = get_embeddings(face)
    prediction = model_svc.predict([embeding])
    return prediction

predictions = []
samples = [os.path.basename(img_path) for img_path in glob.glob(test_path + '*')]
samples.sort()
for sample in samples:
    filename = glob.glob(test_path + sample)[0]
    with Image.open(filename) as img:
        img = img.resize(IMG_SHAPE, Image.ANTIALIAS)
        predictions.append(predict(img))


# %% [markdown]
# Save Predictions

savemat('outputLabel1.mat', {'outputLabel1':predictions})

print('* Done *')