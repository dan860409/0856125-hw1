import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings 
warnings.filterwarnings("ignore")

from keras.preprocessing.image import ImageDataGenerator
from models import build_VGG16, build_VGG19

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

IMG_SHAPE = (256, 256)
train_dir = 'output/train'
validation_dir = 'output/val'
test_dir = 'dataset/test'

train_datagen = ImageDataGenerator(
    # rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    # rescale=1./255,
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True
)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=IMG_SHAPE,
    batch_size=64,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=IMG_SHAPE,
    batch_size=64,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=IMG_SHAPE,
)

print(train_generator.class_indices)
print(validation_generator.class_indices)


# train model and validate
model = build_VGG19()
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
print("\nGet Model !!\n")

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=100,
    epochs=8,
    validation_data=validation_generator,
    validation_steps=50
)


loss, acc = model.evaluate_generator(validation_generator, steps=30)
print(f'\nThe Accuracy on Test Data = {acc}\n')


# predict
import numpy as np
import pandas as pd
from keras.preprocessing import image

df = pd.read_csv("sameple_submission.csv")
class_dict = train_generator.class_indices

for dirname, _, filenames in os.walk(test_dir):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        print(filepath) 

        img = image.load_img(filepath, target_size=IMG_SHAPE)
        img_arr = image.img_to_array(img)
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)

        preds = model.predict(img_arr)
        result_c = list(class_dict.keys())[list(class_dict.values()).index(np.argmax(preds))]
        df.loc[int(filename[6:10]), 'label'] = result_c

df.to_csv('Result.csv')


