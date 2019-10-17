from keras.applications.vgg16 import VGG16
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten


IMG_SHAPE = (256, 256, 3)
label_num = 13

def build_VGG16():
	print('buiding model...\n')

	base_model = VGG16(weights='imagenet', include_top=False, classes=label_num, input_shape=IMG_SHAPE)
	model = Sequential()
	model.add(base_model)

	model.add(Flatten())

	model.add(Dense(256))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	# model.add(Dense(128))
	# model.add(BatchNormalization())
	# model.add(Activation('relu'))
	# model.add(Dropout(0.4))
	model.add(Dense(13))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))

	model.summary()

	# base_model.trainable = True
	# set_trainable = False

	# model.save('VGG19_pretrain_all.model')
	print('return model...\n')
	return model


def build_VGG19():
	base_model = VGG19(include_top=False, weights='imagenet', classes=label_num, input_shape=IMG_SHAPE)
	model = Sequential()
	model.add(base_model)
	model.add(Flatten())

	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(13, activation='softmax'))

	
	model.layers[0].trainable = False
	model.summary()

	# base_model.trainable = True
	# set_trainable = False

	# model.save('VGG19_pretrain_all.model')
	return model
