from sklearn.datasets import load_files

from keras import applications
from keras.utils import np_utils
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.applications import xception

from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pandas as pd
import numpy as np
import glob
import os

test_dir = 'C:/Users/Mahtab Noor Shaan/PycharmProjects/dog_breed_recognition/test'

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

model = xception.Xception(weights='imagenet', include_top=False)

files = glob.glob('C:/Users/Mahtab Noor Shaan/PycharmProjects/dog_breed_recognition/test1/*.jpg')
tensors = paths_to_tensor(files)
data = xception.preprocess_input(tensors)

bottleneck_features = model.predict(data, batch_size=10)

print('Xception test1 bottleneck features shape: {} size: {:,}'.format(bottleneck_features.shape, bottleneck_features.size))

#np.save('bottleneck_features/test1/xception.npy', bottleneck_features)
np.save(open('test1_x_bf_xception.npy', 'wb'), bottleneck_features)