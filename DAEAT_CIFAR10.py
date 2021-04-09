###########################################################################################################
# Code inspired by (last accessed: 09.04.2021)
# https://codahead.com/blog/a-denoising-autoencoder-for-cifar-datasets
# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
# and tensorflow tutorials
###########################################################################################################
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, datasets, models
from tensorflow.keras.models import Model
########################################################
# Helper functions
##################
# Adding random noise to data
def add_noise_to_data (data):
	size = data.shape
	# Getting noise
	noise = np.random.normal(scale=0.1, size = size)
	# Adding noise
	data = data + noise
	# Formating data
	return np.clip(data, 0.0, 1.0)
# Generating adversarial Images for specific model
def add_adversarial_pattern (images, labels, epsilon, path):
	# getting model
	model = get_NN(path)
	counter = 0
	for image in images:
		# random Value between 0 and epsilon
		epsilon = uniform(0, epsilon) 
		# Generating perturbation
		pertubations = create_adversarial_pattern(model,  tf.convert_to_tensor(np.expand_dims(image, 0)), labels[counter])
		# Changing image with epsilon and the perturbation
		images[counter] = image + epsilon * pertubations 
		counter = counter + 1
	return images
# Creating adversarial pattern
def create_adversarial_pattern (model, input_image, input_label):
	# specific to defined model
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
	# watching input_image
	with tf.GradientTape() as tape:
		tape.watch(input_image)
		prediction = model(input_image)
		loss = loss_object(input_label, prediction)
	# Getting gradient
	gradient = tape.gradient(loss, input_image)
	# Get the sign of the gradients to create the perturbation
	signed_grad = tf.sign(gradient)
	return signed_grad
########################################################

# defining model structure
def define_model ():
	inputs = layers.Input(shape=(32, 32, 3), name='dae_input')

	conv_block1 = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
	conv_block1 = layers.BatchNormalization()(conv_block1)

	conv_block2 = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(conv_block1)
	conv_block2 = layers.BatchNormalization()(conv_block2)

	conv_block3 = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(conv_block2)
	conv_block3 = layers.BatchNormalization()(conv_block3)

	conv_block4 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(conv_block3)
	conv_block4 = layers.BatchNormalization()(conv_block4)

	conv_block5 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(conv_block4)
	conv_block5 = layers.BatchNormalization()(conv_block5)
	
	deconv_block1 = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(conv_block5)
	deconv_block1 = layers.BatchNormalization()(deconv_block1)

	merge = layers.Concatenate()([deconv_block1, conv_block3])
	deconv_block2 = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(merge)
	deconv_block2 = layers.BatchNormalization()(deconv_block2)

	merge = layers.Concatenate()([deconv_block2, conv_block2])
	deconv_block3 = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(merge)
	deconv_block3 = layers.BatchNormalization()(deconv_block3)

	merge = layers.Concatenate()([deconv_block3, conv_block1])
	deconv_block4 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(merge)
	deconv_block4 = layers.BatchNormalization()(deconv_block4)

	final_deconv = layers.Conv2DTranspose(filters=3, kernel_size=3, padding='same')(deconv_block4)
	outputs = layers.Activation('sigmoid')(final_deconv)
 
	model = Model(inputs, outputs)
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
# Creating Autoencoder for CIFAR10, optional saving the trained Autoencoder
def create_train_and_save_Autoencoder (path_to_NN, save_path = ''):
	# Loading and generating images, noisy images and adversarial images
	(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
	# Formating
	train_images = train_images.astype('float32') / 255.0
	test_images = test_images.astype('float32') / 255.0
	# Adding random noise
	train_images_noisy = add_noise_to_data(train_images)
	test_images_noisy = add_noise_to_data(test_images)
	# Generating adversarial images
	train_images_adv = add_adversarial_pattern(train_images, train_labels, 0.5, path_to_NN)
	test_images_adv = add_adversarial_pattern(test_images, test_labels, 0.5, path_to_NN)
	# Linking noisy and adversarial images
	train_images_noisy_adv = np.concatenate((train_images_noisy, train_images_adv), axis=0)
	train_images = np.concatenate((train_images, train_images), axis = 0)
	test_images_noisy_adv = np.concatenate((test_images_noisy, test_images_adv), axis = 0)
	test_images = np.concatenate((test_images, test_images), axis = 0)

	# Defining Autoencoder structure
	autoencoder = define_model()

	# Training
	history = autoencoder.fit(train_images_noisy_adv, train_images,
				epochs=20,
				batch_size=10,
				validation_data=(test_images_noisy_adv, test_images),
				verbose = 1)
	# Saving Autoencoder
	if (save_path != ''):
		autoencoder.save(save_path)
	return autoencoder
# Loading saved Autoencoder
def get_NN (path):
	return models.load_model(path)