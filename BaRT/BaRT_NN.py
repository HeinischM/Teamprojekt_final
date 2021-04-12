import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, save_img

import multiprocessing
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import datasets, layers, models

from io import BytesIO

import cifar10_NN as cifar10
import fashionMNIST_NN as fmnist

from BaRT import *



def train_NN_Save(num_epochs= 100, load_clean_model=True, epochs_per_set=100, dataset= 'CIFAR10'):
	if(dataset =='CIFAR10'):
		(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

		train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
		test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)

		train_labels = keras.utils.to_categorical(train_labels, 10)
		test_labels = keras.utils.to_categorical(test_labels, 10)

		class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

	elif(dataset =='F-MNIST'):

		(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

		train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
		test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

		train_labels = keras.utils.to_categorical(train_labels, 10)
		test_labels = keras.utils.to_categorical(test_labels, 10)

		class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

		mean = np.mean(train_images)
		std = np.std(train_images)
		train_images = (train_images - mean) / std

	if (not load_clean_model):
		if(dataset =='CIFAR10'):
			model = cifar10.define_model(True)
		elif(dataset =='F-MNIST'):
			model = fmnist.define_model(True)
		datagen = ImageDataGenerator(rotation_range=15,
				horizontal_flip=True,
				width_shift_range=0.1,
				height_shift_range=0.1,
				zoom_range=0.3)
		datagen.fit(train_images)

		# Training
		print('Model training is starting...')
		history = model.fit(datagen.flow(train_images, train_labels, batch_size=128),
							steps_per_epoch = len(train_images) / 128, 
							epochs=num_epochs,
							validation_data=(test_images, test_labels),
							verbose = 1)
		print('Basic Model training is done!')
		model.save('Clean-Model_'+ str(num_epochs) + '_' + dataset)
	else:
		model = keras.models.load_model('Clean-Model_'+ str(num_epochs) + '_' + dataset)

	num_of_sets =int(num_epochs/epochs_per_set)

	for i in range(num_of_sets):
		if(dataset =='CIFAR10'):
			(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
			train_images = np.load(dataset + str(i)+ "_bart-3_FM" +".npy", allow_pickle=True)
			test_images = np.load(dataset + str(i) + "_bart_val-3_FM"+ ".npy", allow_pickle=True)
			train_images = train_images.reshape(train_images.shape[0],32, 32, 3)
			test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)
		elif(dataset =='F-MNIST'):
			(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
			train_images = np.load(dataset + str(i)+ "_bart-3_FM" +".npy", allow_pickle=True)
			test_images = np.load(dataset + str(i) + "_bart_val-3_FM"+ ".npy", allow_pickle=True)
			train_images = train_images.reshape(train_images.shape[0],28, 28, 1)
			test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

		mean = np.mean(train_images)
		std = np.std(train_images)
		train_images = (train_images - mean) / std

		mean = np.mean(test_images)
		std = np.std(test_images)
		test_images = (test_images - mean) / std 

			
		train_labels = keras.utils.to_categorical(train_labels, 10)
		test_labels = keras.utils.to_categorical(test_labels, 10)
		class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

		K.set_value(model.optimizer.learning_rate, 0.0001)

		print('Model training is starting...')


		history = model.fit(train_images, train_labels, batch_size=128,
							steps_per_epoch = len(train_images) /128, 
							epochs=epochs_per_set,
							validation_data=(test_images, test_labels),
							verbose = 1)

		model.save('Bart_'+ str(num_epochs) + '_epochen_'+ str(num_epochs/epochs_per_set) +'_sets_'+ dataset)

		
	print('Model training is done!')
