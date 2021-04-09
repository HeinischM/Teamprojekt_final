import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

class_names = ['TshirtTop', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'AnkleBoot']
########################################################
# Neural network (NN)
# Initalizes and creates NN (with default Parameters)
def create_and_get_NN (number_of_epochs = 10, verbose = 1, model_save_path = ''):
	# Loading images
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()    

	#Formating
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
	test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

	train_images = train_images.astype('float32')
	test_images = test_images.astype('float32')
	mean = np.mean(train_images)
	std = np.std(train_images)
	test_images = (test_images - mean) / std
	train_images = (train_images - mean) / std

	train_labels = keras.utils.to_categorical(train_labels, 10)
	test_labels = keras.utils.to_categorical(test_labels, 10)

	train_labels = train_labels.astype('int')
	test_labels = test_labels.astype('int')
	
	# Getting the model
	model = define_model()

	# Data augmentation for better image accuracy
	datagen = ImageDataGenerator(rotation_range=15,
			horizontal_flip=True,
			width_shift_range=0.1,
			height_shift_range=0.1,
			zoom_range=0.3)
	datagen.fit(train_images)

	# Training
	history = model.fit(datagen.flow(train_images, train_labels, batch_size=128),
							steps_per_epoch = len(train_images) / 128, 
							epochs=number_of_epochs,
							validation_data=(test_images, test_labels),
							verbose = verbose)
	
	if (model_save_path != ''):    
		model.save(model_save_path)	
	return model
# Defines the model architecture (layers, number of layers, metrics,...)
def define_model ():		
	# Defining the model
		
	model = models.Sequential()
	model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(28, 28, 1),padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.Conv2D(filters=32, kernel_size=3 , activation='relu', padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D(pool_size=2))   
	model.add(layers.Dropout(rate=0.2))

	model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D(pool_size=2))  
	model.add(layers.Dropout(rate=0.2))

	model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D(pool_size=2))  
	model.add(layers.Dropout(rate=0.2))

	model.add(layers.Flatten())
	model.add(layers.Dense(units=512, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(units=10, activation='softmax'))

	# Compile the model
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=Adam(lr=0.001, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
	return model
########################################################
# Adversarial Images --> FGSM
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
# Generating an adversarial Image with FGSM --> DEF: X_avd = X_real + epsilon*perturbation
def create_and_get_adversarial_image (model, img, img_label_index, display_adV_image = False, epsilon = 0.001):
	# Getting prediction of clean image
	img_predictions = model.predict(np.expand_dims(img, 0))
	img_max_prediction_index = np.argmax(img_predictions)
	img_prediction_label = class_names[img_max_prediction_index]

	# Calculating perturbation
	pertubations = create_adversarial_pattern(model, img, img_label_index)

	adv_found = False
	adv_img_prediction_label = None
	# Calculating epsilon
	while((not adv_found) and epsilon <= 1):
		# Creating potential adversarial Image
		adv_img = img + epsilon * pertubations
		adv_img = tf.clip_by_value(adv_img, -1, 1) # Values between -1 and 1

		# Getting prediction of potential adversarial image
		adv_img_predictions = model.predict(adv_img)
		adv_img_prediction_label = class_names[np.argmax(adv_img_predictions)]
		# If not an adversarial Image
		if (adv_img_prediction_label == img_prediction_label):
			# Adding a small value to epsilon
			epsilon = epsilon + 0.001
		else:
			adv_found = True
			if (display_adV_image):
				# Vergleich der Bilder
				plt.figure(figsize=(10,10))
				# clean Image
				plt.subplot(2,1,1)
				plt.imshow(img * 0.5 + 0.5)
				plt.title('Image: {} : {:.2f}% Confidence' .format(img_prediction_label, img_predictions[0][img_max_prediction_index] * 100))
				# Adversarial Image
				plt.subplot(2,1,2)
				plt.imshow(adv_img[0] * 0.5 + 0.5)
				plt.title('Adv_Image: {} : {:.2f}% Confidence' .format(adv_img_prediction_label, adv_img_predictions[0][adv_img_max_prediction_index] * 100))
	
				plt.show()

	return adv_found, adv_img, adv_img_prediction_label
########################################################
# Loading saved NN
def get_NN (path):
	return models.load_model(path)