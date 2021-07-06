# %% [markdown]
# Set Variables
IMG_SHAPE = (224, 224)
ITER_SIZE = 5
BATCH_SIZE = 8

import sys
if len(sys.argv) < 2:
    print('Usage: python FaceRecognition2.py <train_directory> <test_directory>')
    sys.exit(1)
train_path = sys.argv[1]
test_path = sys.argv[2]

# %% [markdown]
# Load Dependencies
print('* Importing dependencies *')

import os
import glob
import numpy as np
import itertools
from PIL import Image, ImageOps
from scipy.io import savemat
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# %% [markdown]
# Load Train Dataset
print('* Augumenting train samples *')

data_gen = ImageDataGenerator(rescale = 1./255, rotation_range = 10, zoom_range = 0.1,
                              width_shift_range = 0.15, height_shift_range = 0.15,
                              brightness_range = [0.2, 1.0], fill_mode = 'nearest')

def augument_2(img):
    img = np.expand_dims(img, 0)
    it = data_gen.flow(img)
    return np.squeeze(it.next(), 0)

def augument_1(img):
    img_resized = img.resize(IMG_SHAPE, Image.ANTIALIAS)
    img_flipped = ImageOps.mirror(img_resized)
    augments = []
    augments.append(np.asarray(img_resized))
    augments.append(np.asarray(img_flipped))
    for i in range(len(augments)): augments.append(augument_2(augments[i]))
    return augments

train_x = []
train_y = [os.path.basename(img_path) for img_path in glob.glob(train_path + '*')]
label_len = len(train_y[0])
for label in train_y:
    filename = glob.glob(train_path + label + '/*')[0]
    with Image.open(filename).convert('RGB') as img:
        augments = augument_1(img)
        train_x.append(augments)

# %% [markdown]
# Load Model
print('* Loading VGG16 model *')

class VGG16(Sequential):
    def layer(self, kernel_no):
        if kernel_no <= 128:
            self.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
            self.add(Convolution2D(kernel_no, (3, 3), activation='relu'))
            self.add(ZeroPadding2D((1,1)))
            self.add(Convolution2D(kernel_no, (3, 3), activation='relu'))
            self.add(MaxPooling2D((2,2), strides=(2,2)))
        elif kernel_no >= 1024:
            self.add(Convolution2D(kernel_no, (7, 7), activation='relu'))
            self.add(Dropout(0.5))
            self.add(Convolution2D(kernel_no, (1, 1), activation='relu'))
            self.add(Dropout(0.5))
            self.add(Convolution2D(2622, (1, 1)))
            self.add(Flatten())
        else:
            self.add(ZeroPadding2D((1,1)))
            self.add(Convolution2D(kernel_no, (3, 3), activation='relu'))
            self.add(ZeroPadding2D((1,1)))
            self.add(Convolution2D(kernel_no, (3, 3), activation='relu'))
            self.add(ZeroPadding2D((1,1)))
            self.add(Convolution2D(kernel_no, (3, 3), activation='relu'))
            self.add(MaxPooling2D((2,2), strides=(2,2)))
        return self
    
    def weight(self, weights_path):
        self.load_weights(weights_path)
        return self

    def freeze(self):
        for layer in self.layers:
            layer.trainable = False
        return self

model = VGG16().layer(64).layer(128).layer(256).layer(512).layer(512).layer(4096)
model = model.weight('vgg_face_weights.h5').freeze()
out = Dense(512, activation='relu', name='fc3')(model.layers[-1].output)
out = Dense(100, activation='softmax', name='prediction')(out)
model = Model(inputs=model.input, outputs=out)
model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=Adam(lr=0.001))

# %% [markdown]
# Encode for Training
print('* Encoding train data *')

train_y = np.repeat(train_y, len(train_x[0]))
train_x = list(itertools.chain.from_iterable(train_x))

label_encoder = LabelEncoder()
categ_encoder = OneHotEncoder(sparse=False)
train_y = label_encoder.fit_transform(train_y)
train_y = train_y.reshape(len(train_y), 1)
train_y = categ_encoder.fit_transform(train_y)

# %% [markdown]
# Train Model
print('* Training VGG16 model *')

model.fit(np.array(train_x), np.array(train_y), epochs = ITER_SIZE, batch_size = BATCH_SIZE, shuffle=True,
                            callbacks = [ReduceLROnPlateau(monitor="loss", factor=0.8, patience=2, verbose=True)])

# %%
del train_x
del train_y

# %% [markdown]
# Predict Samples
print('* Predicting test samples *')

def predict(img):
    prediction = model.predict(np.expand_dims(img, axis=0))
    prediction_index = np.argmax(prediction)+1
    prediction_label = str(prediction_index).zfill(label_len)
    return prediction_label

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

savemat('outputLabel2.mat', {'outputLabel2':predictions})

print('* Done *')